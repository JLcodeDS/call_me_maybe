"""Finite State Machine (FSM) for Pydantic-driven structural token validation.

This module implements a grammar-based parsing state machine that evaluates
partial SON response strings against a Pydantic schema to enforce structured
JSON token generation at runtime.
"""

from pydantic import BaseModel


class PydanticFSM:
    """Finite State Machine to govern structured JSON output matching a
    Pydantic model.

    Tracks JSON parsing states (inside strings, scanning object keys, waiting
    for colons, processing array brackets) to filter next-token options via a
    pre-compiled token vocabulary cache.

    Attributes:
        param_model: The Pydantic model type used to extract validation
            boundaries.
        param_json: The complete raw dictionary representing the model's
            JSON schema.
        token_cache: A dictionary tracking pre-filtered token ID sets indexed
            by syntactic category names.
        properties: The specific property definitions mapping schema keys to
            expected types.
    """

    def __init__(self, param_model: type[BaseModel],
                 token_cache: dict[str, set[int]]):
        """Initializes PydanticFSM and extracts properties from the Pydantic
        schema.

        Args:
            param_model: The reference Pydantic validation structure class.
            token_cache: Categorized sets of integer vocabulary tokens.
        """
        self.param_model = param_model
        self.param_json = param_model.model_json_schema()
        self.token_cache = token_cache
        self.properties = self.param_json['$defs'].get(
            'parameters').get('properties')

    def get_allowed_tokens(self, response_buffer: str,
                           entire_response_buffer: str) -> set[int] | None:
        """Determines valid token ID groups based on the current JSON text
        parsing state.

        Parses the sequential layout of the current buffer to detect character
        boundaries, parameter schemas, and JSON syntax blocks, matching them
        to Allowed Vocabulary masks.

        Args:
            response_buffer: The current text block being parsed for the active
                parameter segment.
            entire_response_buffer: The complete accumulated generation context
                from the start.

        Returns:
            A set of integer token IDs allowed to follow the current buffer
            layout, or None if the entire JSON structure is fully closed and
            valid.
        """
        if self.has_closed_brackets(
                entire_response_buffer) and entire_response_buffer.strip(
                    ).endswith("}"):
            return None

        if not response_buffer:
            return self.token_cache["START_PARAM"]

        in_string = False
        current_key = ""
        building_key = False
        expect_colon = False
        expect_value = False
        expect_comma_or_end = False
        is_escaped = False
        array_depth = 0
        i = 0

        while i < len(response_buffer):
            char = response_buffer[i]
            if in_string and not is_escaped and char == '\\':
                is_escaped = True
                i += 1
                continue
            if is_escaped:
                is_escaped = False
                i += 1
                continue
            if char == '"':
                in_string = not in_string
                if in_string:
                    if not expect_value and not expect_comma_or_end:
                        building_key = True
                        current_key = ""
                else:
                    if building_key:
                        building_key = False
                        expect_colon = True
                    elif expect_value:
                        expect_value = False
                        expect_comma_or_end = True
            elif not in_string:
                if char == '[':
                    array_depth += 1
                    expect_value = True
                elif char == ']':
                    array_depth -= 1
                    expect_value = False
                    expect_comma_or_end = True
                elif char == '{':
                    pass
                elif char == ':':
                    expect_colon = False
                    expect_value = True
                elif char == ',':
                    expect_comma_or_end = False
                    expect_value = False
                    if array_depth > 0:
                        expect_value = True
                elif building_key:
                    current_key += char
            elif in_string and building_key:
                current_key += char
            i += 1
        if in_string:
            return self.token_cache["STRING_INNER"]
        if expect_colon:
            return self.token_cache["EXPECT_COLON"]
        if expect_value:
            param_schema = self.properties.get(current_key, {})
            is_array_type = (param_schema.get("type") == "array")
            if is_array_type and array_depth == 0:
                return self.token_cache["ARR_START_END"]
            if is_array_type and array_depth > 0:
                param_type = param_schema.get("items", {}).get("type")
            else:
                param_type = param_schema.get("type")
            if param_type in ("number", "integer"):
                allowed = self.token_cache["NUMBER_CONTINUE"].union(
                    self.token_cache["NEXT_DELIMITER"])
                if array_depth > 0:
                    allowed = allowed.union(self.token_cache["ARR_START_END"])
                return allowed
            if param_type in ("boolean", "null"):
                allowed = self.token_cache["BOOL_CONTINUE"].union(
                    self.token_cache["NEXT_DELIMITER"])
                if array_depth > 0:
                    allowed = allowed.union(self.token_cache["ARR_START_END"])
                return allowed
            if param_type == "string":
                return self.token_cache["ANY_VALUE_START"]
        if expect_comma_or_end:
            allowed = self.token_cache["NEXT_DELIMITER"]
            if array_depth > 0:
                allowed = allowed.union(self.token_cache["ARR_START_END"])
            return allowed
        return self.token_cache["ANY_VALUE_START"]

    @staticmethod
    def has_closed_brackets(s: str) -> bool:
        """Validates if all braces and brackets within a string are correctly
        paired and closed.

        Processes characters sequentially while respecting backslash-escapes
        and raw literal string boundaries to ignore syntax indicators embedded
        within JSON values.

        Args:
            s: The full text string to evaluate.

        Returns:
            True if all opened characters ('{', '[', '(') match their
            respective closing markers, False otherwise.
        """
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
