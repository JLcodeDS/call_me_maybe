from .data_models import TokenTrie
from llm_sdk import Small_LLM_Model  # type:ignore
from typing import Any
from pydantic import BaseModel


class PydanticFSM:
    def __init__(self, param_model: type[BaseModel],
                 token_cache: dict[str, Any]):
        self.param_model = param_model
        self.param_json = param_model.model_json_schema()
        self.token_cache = token_cache
        self.struct_stack: list[str] = []
        self.next_state = "START_PARAM"

    def get_allowed_tokens(self, current_response: str) -> set[int] | None:
        in_string, last_struct = self.get_str_context(current_response)
        if in_string:
            return self.token_cache["STRING_INNER"]
        if last_struct == '"' and self.is_closing_key(current_response):
            return self.token_cache["EXPECT_COLON"]
        if last_struct == '"':
            return self.token_cache["NEXT_DELIMITER"]
        if last_struct == ":":
            return self.token_cache["ANY_VALUE_START"]
        if last_struct == ",":
            return self.token_cache["ANY_VALUE_START"]
        if current_response and (
                current_response[-1].isdigit() or current_response[-1] == '.'):
            return self.token_cache["NUMBER_CONTINUE"]
        if self.has_valid_brackets(current_response):
            return None
        return self.token_cache["PARAM_START"]

    def is_closing_key(self, current_response: str) -> bool:
        response = current_response[:-1]
        for i, j in enumerate(response[::-1]):
            if j == '"':
                break
        return (response[i:] in self.param_json["required"])

    def get_str_context(self, current_response: str) -> tuple[bool, str]:
        in_string = False
        is_escaped = False
        last_structural_char = ""
        for char in current_response:
            if is_escaped:
                is_escaped = False
                continue
            if char == '\\':
                is_escaped = True
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
    def has_valid_brackets(s: str) -> bool:
        bracks = {
            "(": ")",
            "[": "]",
            "{": "}",
        }
        current_stack = []
        for c in s:
            if c in bracks.keys():
                current_stack.append(c)
            if c in bracks.values() and (
                c == bracks[current_stack[-1]]
            ):
                current_stack.pop()
        if len(current_stack):
            return False
        return True

# class FSM(ABC):
#     def __init__(
#             self, token_cache: dict[str, set[int]], model: Small_LLM_Model):
#         self.token_cache = token_cache
#         self.model = model

#     @abstractmethod
#     def get_allowed_next_tokens(
#             self, current_tokens: list[int]) -> set[int] | None:
#         ...

#     @abstractmethod
#     def is_end_state(self, current_tokens: list[int]) -> bool:
#         ...


# class BoolTrie(FSM):
#     def __init__(
#             self, token_cache: dict[str, set[int]], model: Small_LLM_Model):
#         super().__init__(token_cache, model)
#         self.boolTrie = TokenTrie()
#         self.insert_true_false()

#     def get_allowed_next_tokens(
#             self, current_tokens: list[int]) -> set[int] | None:
#         return self.boolTrie.get_next_allowed_tokens(current_tokens)

#     def is_end_state(self, current_tokens: list[int]) -> bool:
#         if self.boolTrie.get_next_allowed_tokens(current_tokens) is None:
#             return True
#         return False

#     def insert_true_false(self) -> None:
#         t_enc = self.model.encode("true")[0].tolist()
#         f_enc = self.model.encode("false")[0].tolist()
#         self.boolTrie.add_token_branch(t_enc)
#         self.boolTrie.add_token_branch(f_enc)


# class IntFSM(FSM):
#     def is_end_state(self, current_tokens: list[int]) -> bool:
#         return True


# class NumFSM(FSM):
#     def is_end_state(self, current_tokens: list[int]) -> bool:
#         return True


# class StringFSM(FSM):
#     def __init__(
#                 self, token_cache: dict[str, set[int]],
#                 model: Small_LLM_Model):
#         super().__init__(token_cache, model)
#         self.state = "START_STR"

#     def is_end_state(self, current_tokens: list[int]) -> bool:
#         if not current_tokens:
#             return False
#         decoded = self.model.decode(current_tokens)
#         decoded_clean = decoded.replace("Ġ", " ")
#         if decoded_clean.startswith('"') and len(decoded_clean) > 1:
#             if decoded_clean.endswith(
#                     '"') and not decoded_clean.endswith('\\"'):
#                 return True
#         return False

#     def get_allowed_next_tokens(self, current_tokens) -> set[int]:
#         if not current_tokens:
#             return self.token_cache["START_STR"]
#         allowed_tokens = self.token_cache["INSIDE_STR"].union(
#             self.token_cache["END_STR"])
#         return allowed_tokens


# class FSMDispatch:
#     def __init__(self, param_type: str,
#                  token_cache: dict[str, set[int]],
#                  model: Small_LLM_Model):
#         self.forward_fsm = {
#             "boolean": BoolTrie,
#             "integer": IntFSM,
#             "string": StringFSM,
#             "number": NumFSM
#         }
#         self.param_type = param_type
#         self.token_cache = token_cache
#         self.model = model

#     def dispatch(self) -> FSM:
#         return self.forward_fsm[self.param_type](self.token_cache, self.model)
