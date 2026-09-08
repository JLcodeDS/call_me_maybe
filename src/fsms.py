from typing import Any
from pydantic import BaseModel


class PydanticFSM:
    def __init__(self, param_model: type[BaseModel],
                 token_cache: dict[str, Any]):
        self.param_model = param_model
        self.param_json = param_model.model_json_schema()
        self.token_cache = token_cache
        self.struct_stack: list[str] = []
        self.required_keys = self.param_json["required"]

    def get_allowed_tokens(self, response_buffer: str) -> set[int] | None:
        in_string, last_struct = self.get_str_context(response_buffer)
        if in_string:
            return self.token_cache["STRING_INNER"]
        if last_struct == '{':
            return self.token_cache["STRING_INNER"]
        if last_struct == '"' and self.is_closing_key(response_buffer):
            return self.token_cache["EXPECT_COLON"]
        if last_struct == '"':
            return self.token_cache["NEXT_DELIMITER"]
        if last_struct == ":":
            return self.token_cache["ANY_VALUE_START"]
        if last_struct == ",":
            return self.token_cache["ANY_VALUE_START"]
        if response_buffer and (
                response_buffer[-1].isdigit() or response_buffer[-1] == '.'):
            return self.token_cache["NUMBER_CONTINUE"].union(
                self.token_cache["NEXT_DELIMITER"])
        if self.has_closed_brackets(response_buffer):
            return None
        return self.token_cache["START_PARAM"]

    def is_closing_key(self, current_response: str) -> bool:
        response = current_response[:-1]
        start_quote_idx = response.rfind('"')
        if start_quote_idx == -1:
            return False
        extracted_key = response[start_quote_idx + 1:]
        return extracted_key in self.required_keys

    def get_str_context(self, current_response: str) -> tuple[bool, str]:
        in_string = False
        last_structural_char = ""
        length = len(current_response)
        i = length - 1
        while i >= 0 and current_response[i] == '\\':
            i -= 1
        for char in current_response:
            if char == '\\':
                continue
            if char == '"':
                in_string = not in_string
                last_structural_char = '"'
                continue
            if not in_string:
                if char in ('{', '}', '[', ']', ':', ','):
                    last_structural_char = char
        return in_string, last_structural_char

    @staticmethod
    def has_closed_brackets(s: str) -> bool:
        if not s:
            return False
        current_stack = []
        for c in s:
            if c in ('(', '[', '{'):
                current_stack.append(c)
            if not current_stack:
                return True
            elif c == ')':
                if current_stack[-1] != '(':
                    return False
            elif c == ']':
                if current_stack.pop() != '[':
                    return False
            elif c == '}':
                if current_stack.pop() != '{':
                    return False
        return len(current_stack) == 0
