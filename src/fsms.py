from .data_models import TokenTrie
from llm_sdk import Small_LLM_Model  # type:ignore
from abc import ABC, abstractmethod


class FSM(ABC):
    def __init__(
            self, token_cache: dict[str, set[int]], model: Small_LLM_Model):
        self.token_cache = token_cache
        self.model = model

    @abstractmethod
    def get_allowed_next_tokens(
            self, current_tokens: list[int]) -> set[int] | None:
        ...

    @abstractmethod
    def is_end_state(self, current_tokens: list[int]) -> bool:
        ...


class BoolTrie(FSM):
    def __init__(
            self, token_cache: dict[str, set[int]], model: Small_LLM_Model):
        super().__init__(token_cache, model)
        self.boolTrie = TokenTrie()
        self.insert_true_false()

    def get_allowed_next_tokens(
            self, current_tokens: list[int]) -> set[int] | None:
        return self.boolTrie.get_next_allowed_tokens(current_tokens)

    def is_end_state(self, current_tokens: list[int]) -> bool:
        if self.boolTrie.get_next_allowed_tokens(current_tokens) is None:
            return True
        return False

    def insert_true_false(self) -> None:
        t_enc = self.model.encode("true")[0].tolist()
        f_enc = self.model.encode("false")[0].tolist()
        self.boolTrie.add_token_branch(t_enc)
        self.boolTrie.add_token_branch(f_enc)


class IntFSM(FSM):
    def is_end_state(self, current_tokens: list[int]) -> bool:
        return True


class NumFSM(FSM):
    def is_end_state(self, current_tokens: list[int]) -> bool:
        return True


class StringFSM(FSM):
    def __init__(
                self, token_cache: dict[str, set[int]],
                model: Small_LLM_Model):
        super().__init__(token_cache, model)
        self.state = "START_STR"

    def is_end_state(self, current_tokens: list[int]) -> bool:
        if not current_tokens:
            return False
        decoded = self.model.decode(current_tokens)
        decoded_clean = decoded.replace("Ġ", " ")
        if decoded_clean.startswith('"') and len(decoded_clean) > 1:
            if decoded_clean.endswith(
                    '"') and not decoded_clean.endswith('\\"'):
                return True
        return False

    def get_allowed_next_tokens(self, current_tokens) -> set[int]:
        if not current_tokens:
            return self.token_cache["START_STR"]
        allowed_tokens = self.token_cache["INSIDE_STR"].union(
            self.token_cache["END_STR"])
        return allowed_tokens


class FSMDispatch:
    def __init__(self, param_type: str,
                 token_cache: dict[str, set[int]],
                 model: Small_LLM_Model):
        self.forward_fsm = {
            "boolean": BoolTrie,
            "integer": IntFSM,
            "string": StringFSM,
            "number": NumFSM
        }
        self.param_type = param_type
        self.token_cache = token_cache
        self.model = model

    def dispatch(self) -> FSM:
        return self.forward_fsm[self.param_type](self.token_cache, self.model)
