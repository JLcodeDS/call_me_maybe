from pydantic import BaseModel


class PydanticFSM:
    def __init__(self, param_model: type[BaseModel],
                 token_cache: dict[str, set[int]]):
        self.param_model = param_model
        self.param_json = param_model.model_json_schema()
        self.token_cache = token_cache
        self.properties = self.param_json['$defs'].get(
            'parameters').get('properties')
    # =========================================================================
    # CORE REFACTOR: Shift from loose lookback guessing to a clean linear state scanner
    # =========================================================================
    
    def get_allowed_tokens(self, response_buffer: str, entire_response_buffer: str) -> set[int] | None:
        """
        Parses the text buffer from left to right to discover the exact syntactical
        state and the currently active parameter name.
        """
        # Clean terminal exit: If the structure is completely closed, terminate generation
        if self.has_closed_brackets(
                entire_response_buffer) and entire_response_buffer.strip(
                    ).endswith("}"):
            return None

        if not response_buffer:
            return self.token_cache["START_PARAM"]

        in_string = False
        current_key = ""
        building_key = False
        
        # State tracking flags
        expect_colon = False
        expect_value = False
        expect_comma_or_end = False

        # --- NEW: Step-by-step state tracker loop ---
        i = 0
        while i < len(response_buffer):
            char = response_buffer[i]
            
            # Handle escape sequences safely so they don't break string boundaries
            if char == '\\' and i + 1 < len(response_buffer):
                i += 2
                continue
                
            if char == '"':
                in_string = not in_string
                if in_string:
                    # If we aren't expecting a value or delimiter, we must be opening a key
                    if not expect_value and not expect_comma_or_end:
                        building_key = True
                        current_key = ""
                else:
                    # String just closed
                    if building_key:
                        building_key = False
                        expect_colon = True
                    elif expect_value:
                        expect_value = False
                        expect_comma_or_end = True
            elif not in_string:
                if char == '{':
                    pass  # Outer JSON wrapper opened
                elif char == ':':
                    expect_colon = False
                    expect_value = True
                elif char == ',':
                    expect_comma_or_end = False
                    expect_value = False
                elif char in ('}', ']'):
                    pass
                elif building_key:
                    current_key += char
            elif in_string and building_key:
                current_key += char
                
            i += 1

        # =========================================================================
        # TARGETED TOKEN MASKING: Return token groups bound directly to state rules
        # =========================================================================
        # print(response_buffer)
        # print(f"str: {in_string}, colon: {expect_colon}, value: {expect_value}, comma_end: {expect_comma_or_end}")
        if in_string:
            return self.token_cache["STRING_INNER"]
            
        if expect_colon:
            return self.token_cache["EXPECT_COLON"]
            
        if expect_value:
            # --- NEW: Dynamic type enforcement based on active key schema ---
            param_type = self.properties.get(current_key, {}).get("type")
            if param_type == "number" or param_type == "integer":
                # CRITICAL FIX: Strip quote boundaries out entirely for numbers.
                # Force the model to output a raw mathematical digits block.
                return self.token_cache["NUMBER_CONTINUE"].union(
                    self.token_cache["NEXT_DELIMITER"])
                
            if param_type == "boolean":
                # Force exact literal true/false tokens immediately
                return self.token_cache["BOOL_CONTINUE"].union(
                    self.token_cache["NEXT_DELIMITER"])
                
            # Default fallback for strings
            return self.token_cache["ANY_VALUE_START"]
            
        if expect_comma_or_end:
            return self.token_cache["NEXT_DELIMITER"]

        return self.token_cache["ANY_VALUE_START"]

    @staticmethod
    def has_closed_brackets(s: str) -> bool:
        if not s:
            return False
        current_stack = []
        in_string = False
        i = 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s):
                i += 2
                continue
            if s[i] == '"':
                in_string = not in_string
            elif not in_string:
                if s[i] in ('(', '[', '{'):
                    current_stack.append(s[i])
                elif s[i] == '}':
                    if not current_stack or current_stack.pop() != '{':
                        return False
                elif s[i] == ']':
                    if not current_stack or current_stack.pop() != '[':
                        return False
            i += 1
        return len(current_stack) == 0
