from .data_preparation import BuildTokenCache


class BoolFSM:
    def __init__(self, token_cache: dict[str, set[int]]):
        self.bool_tokens = token_cache["BOOL"]

    def get_allowed_tok_from_state(self) -> set[int]:
        return self.bool_tokens

    def is_end_state(self, current: str) -> bool:
        return current == "true" or current == "false"


class IntFSM:
    pass


class NumFSM:
    pass


class StringFSM:
    pass


class FSMDispatch:
    def __init__(self, param_type: str, token_cache: dict[str, set[int]]):
        self.forward_fsm = {
            "bool": BoolFSM,
            "integer": IntFSM,
            "string": StringFSM,
            "number": NumFSM
        }
        self.param_type = param_type
        self.token_cache = token_cache

    def dispatch(self) -> type[object]:
        return self.forward_fsm[self.param_type](self.token_cache)
