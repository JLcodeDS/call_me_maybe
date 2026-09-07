from llm_sdk import Small_LLM_Model  # type: ignore
from .data_models import TokenTrie
from typing import Any, Callable
import re


class PrepareFnData:
    def __init__(self, function_data: list[dict[str, Any]]):
        self.function_data = function_data
        self.fn_names_defs = self.get_fn_names_defs()
        self.fn_params = self.get_fn_params()

    def get_fn_names_defs(self) -> list[dict[str, str]]:
        return [
            {"name": fn["name"], "description": fn["description"]
             } for fn in self.function_data
            ]

    def get_fn_params(self) -> dict[str, Any]:
        return {
            fn["name"]: {
                param: val["type"] for param, val in fn["parameters"].items()
            } for fn in self.function_data
        }


class CreateTries:
    def __init__(self, function_data: list[dict[str, Any]],
                 model: Small_LLM_Model) -> None:
        self.function_data = function_data
        self.function_names = self.get_fn_names()
        self.FnTrie = TokenTrie()
        self.model = model
        self.insert_fns()

    def make_encode(self, func) -> str:
        return self.model.encode(func)[0].tolist()

    def insert_fns(self):
        for func in self.function_names:
            self.FnTrie.add_token_branch(
                        self.make_encode(func)
                        )

    def get_fn_names(self) -> list[str]:
        return [fn["name"] for fn in self.function_data]


class BuildTokenCache:
    def __init__(self, str_to_tok: dict[str, int]) -> None:
        self.str_to_tok = str_to_tok
        self.token_cache: dict[str, Any] = {}

    def filter_tokens(self, f: Callable) -> set[int]:
        return {
                v for k, v in self.str_to_tok.items() if f(k)
            }

    def build_cache(self) -> None:
        self.token_cache["START_PARAM"] = self.filter_tokens(
            lambda s: any(s.startswith('{'))
        )

        self.token_cache["ANY_VALUE_START"] = self.filter_tokens(
            lambda s: any(s.lstrip().startswith(c) for c in [
                '"', '-', 't', 'f', 'n', '[']) or
            bool(re.match(r'^\s*[0-9]', s))
        )

        self.token_cache["NUMBER_CONTINUE"] = self.filter_tokens(
            lambda s: bool(re.match(r'^[0-9eE.+\-]+$', s)) or
            any(s.startswith(c) for c in [',', '}', ']'])
        )
        self.token_cache["STRING_INNER"] = self.filter_tokens(
            lambda s: ('"' not in s or s.startswith('"') or s == '"'
                       ) and not any(
                ord(c) < 32 for c in s)
            )

        self.token_cache["NEXT_DELIMITER"] = self.filter_tokens(
            lambda s: any(c in s for c in [',', '}', ']']) or any(
                s.lstrip().startswith(c) for c in [',', '}', ']'])
        )

        self.token_cache["EXPECT_COLON"] = self.filter_tokens(
            lambda s: ':' in s or s.lstrip().startswith(':')
        )

    # def build_cache(self) -> None:
    #     self.token_cache["START_STATE"] = self.filter_tokens()
    #     self.token_cache["INT"] = self.filter_tokens(self.is_int)
    #     self.token_cache["NUMBER"] = self.filter_tokens(self.is_number_token)
    #     self.token_cache["INSIDE_STR"] = self.filter_tokens(self.isin_str)
    #     self.token_cache["START_STR"] = self.filter_tokens(self.isstart_str)
    #     self.token_cache["END_STR"] = self.filter_tokens(self.isend_str)
    #     self.token_cache["BOOL"] = self.filter_tokens(
    #         lambda s: s in {"true", "false", "tr", "tru", "fa", "fal", "fals"}
    #         )
    #     self.token_cache["END_NUM"] = self.filter_tokens(self.isend_num)
    #     self.token_cache["END_INT"] = self.filter_tokens(self.isend_int)

    

    # @classmethod
    # def start_state(cls, s: str):

    # @staticmethod
    # def clean_spaces(s: str):
    #     #  Might need to change to work for other models
    #     return s.replace('Ġ', ' ')

    # @staticmethod
    # def is_int(s: str):
    #     if s == "-":
    #         return True
    #     try:
    #         int(s)
    #         return True
    #     except ValueError:
    #         return False

    # @classmethod
    # def isend_num(cls, s: str):
    #     if s == ',':
    #         return True
    #     if s.endswith(','):
    #         return cls.is_number_token(s[:-1])
    #     return False

    # @classmethod
    # def isend_int(cls, s: str):
    #     if s == ',':
    #         return True
    #     if s.endswith(','):
    #         return cls.is_int(s[:-1])
    #     return False

    # @staticmethod
    # def is_number_token(s: str):
    #     if s in ("-", "."):
    #         return True
    #     if s.startswith("-"):
    #         s = s.replace("-", "", 1)
    #     if s.replace(".", "", 1).isdigit():
    #         return True
    #     return False

    # @classmethod
    # def isin_str(cls, s: str):
    #     s = cls.clean_spaces(s)
    #     if any(cls.is_control_char(c) for c in s):
    #         return False
    #     i = 0
    #     while i < len(s):
    #         if s[i] == '"':
    #             num_backslashes = 0
    #             j = i - 1
    #             while j >= 0 and s[j] == "\\":
    #                 num_backslashes += 1
    #                 j -= 1
    #             if num_backslashes % 2 == 0:
    #                 return False
    #         i += 1
    #     return True

    # @classmethod
    # def isend_str(cls, s: str):
    #     s = cls.clean_spaces(s)
    #     if s == '"':
    #         return True
    #     if s.endswith('"') and not s.endswith('\\"'):
    #         return cls.isin_str(s[:-1])
    #     return False

    # @classmethod
    # def isstart_str(cls, s: str):
    #     s = cls.clean_spaces(s)
    #     if s == '"':
    #         return True
    #     return False

    # @staticmethod
    # def is_control_char(c: str):
    #     return ord(c) < 32  # ascii num
