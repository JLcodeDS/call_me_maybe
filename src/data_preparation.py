from llm_sdk import Small_LLM_Model  # type: ignore
from .data_models import TokenTrie
from typing import Any, Callable, cast
import re


class PrepareFnData:
    def __init__(self, function_data: list[dict[str, Any]]):
        self.function_data = function_data
        self.fn_names_defs = self.get_fn_names_defs()
        self.fn_params = self.get_fn_params()
        self.json_params = self.get_json_params()

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

    def get_json_params(self) -> dict[str, Any]:
        return {
            fn["name"]: fn["parameters"] for fn in self.function_data
        }


class CreateTries:
    def __init__(self, function_data: list[dict[str, Any]],
                 model: Small_LLM_Model) -> None:
        self.function_data = function_data
        self.function_names = self.get_fn_names()
        self.FnTrie = TokenTrie()
        self.model = model
        self.insert_fns()

    def make_encode(self, func: str) -> list[int]:
        encode = self.model.encode(func)[0].tolist()
        return cast(list[int], encode)

    def insert_fns(self) -> None:
        for func in self.function_names:
            self.FnTrie.add_token_branch(
                        self.make_encode(func)
                        )

    def get_fn_names(self) -> list[str]:
        return [fn["name"] for fn in self.function_data]


class BuildTokenCache:
    def __init__(self, str_to_tok: dict[str, int]) -> None:
        self.str_to_tok = str_to_tok
        self.token_cache: dict[str, set[int]] = {}

    def filter_tokens(self, f: Callable[[str], bool]) -> set[int]:
        return {
                v for k, v in self.str_to_tok.items() if f(k.replace("Ġ", " "))
            }

    def build_cache(self) -> None:
        # 1. Start Parameter Block
        # Matches exactly the opening brace of the parameter JSON object
        self.token_cache["START_PARAM"] = self.filter_tokens(
            lambda s: s.strip().startswith('{')
        )

        # 2. General/String Value Start
        # Used strictly for strings or arbitrary objects. Includes opening quotes.
        self.token_cache["ANY_VALUE_START"] = self.filter_tokens(
            lambda s: any(s.lstrip().startswith(c) for c in ['"', '['])
        )

        # 3. Boolean/Null Literals
        # Allows full or fractional structural pieces of true, false, and null
        self.token_cache["BOOL_CONTINUE"] = self.filter_tokens(
            lambda s: any(s.strip().startswith(c) for c in [
                "t", "tr", "tru", "true",
                "f", "fa", "fal", "fals", "false",
                "n", "nu", "nul", "null"
            ])
        )

        # 4. Strict Number Continuations (CRITICAL UPDATE)
        # Strips out trailing punctuation hooks (, } ]). 
        # Only allows pure numeric notation: digits, signs, decimals, and exponents.
        self.token_cache["NUMBER_CONTINUE"] = self.filter_tokens(
            lambda s: bool(re.match(
                r'^-?[0-9]*\.?[0-9]*([eE][+-]?[0-9]*)?$', s.strip()))
            and s.strip() != ""
        )

        # 5. Inner String Contents
        # Allows any characters except unescaped double quotes or control codes
        self.token_cache["STRING_INNER"] = self.filter_tokens(
            lambda s: ('"' not in s or s.startswith(
                '"') or s == '"' or s.rstrip().endswith('"'))
            and not any(ord(c) < 32 for c in s)
        )

        # 6. Syntax Structural Delimiters
        # Handles transition boundaries separating keys, values, or object endings
        self.token_cache["NEXT_DELIMITER"] = self.filter_tokens(
            lambda s: any(c in s for c in [',', '}', ']']) or
            any(s.lstrip().startswith(c) for c in [',', '}', ']'])
        )

        # 7. Key/Value Structural Map
        # Strictly matches the structural mapping separator colon
        self.token_cache["EXPECT_COLON"] = self.filter_tokens(
            lambda s: ':' in s or s.lstrip().startswith(':')
        )
