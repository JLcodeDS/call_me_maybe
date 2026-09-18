"""Data preparation, prefix tree generation, and token caching utilities.

This module provides classes to process raw function definitions, build
token-level prefix trees (Tries) for constrained generation, and
pre-compute token caches categorized by syntactic JSON patterns.
"""

from llm_sdk import Small_LLM_Model  # type: ignore
from .data_models import TokenTrie
from typing import Any, Callable, cast
import re


class PrepareFnData:
    """Processes and structures raw function metadata for the validation
    pipeline.

    Attributes:
        function_data: A list of dictionaries containing raw function
            definitions.
        fn_names_defs: A list of mappings extracting only the name and
            description
            of each function.
        fn_params: A dictionary mapping function names to a simplified
            key-type layout.
        json_params: A dictionary mapping function names to their full
            parameter schemas.
    """

    def __init__(self, function_data: list[dict[str, Any]]):
        """Initializes PrepareFnData and extracts components from the input
        data.

        Args:
            function_data: Raw specification structures for available
                functions.
        """
        self.function_data = function_data
        self.fn_names_defs = self.get_fn_names_defs()
        self.fn_params = self.get_fn_params()
        self.json_params = self.get_json_params()

    def get_fn_names_defs(self) -> list[dict[str, str]]:
        """Extracts names and text descriptions from all functions.

        Returns:
            A list of dictionaries where each item contains "name" and
                "description" keys.
        """
        return [
            {"name": fn["name"], "description": fn["description"]
             } for fn in self.function_data
            ]

    def get_fn_params(self) -> dict[str, Any]:
        """Maps function names to a flat dictionary of parameter types.

        Returns:
            A dictionary where keys are function names and values are
                dictionaries mapping parameter names to their type strings.
        """
        return {
            fn["name"]: {
                param: val["type"] for param, val in fn["parameters"].items()
            } for fn in self.function_data
        }

    def get_json_params(self) -> dict[str, Any]:
        """Maps function names directly to their structural parameters
        definitions.

        Returns:
            A dictionary where keys are function names and values are the
            complete raw parameters sub-schemas.
        """
        return {
            fn["name"]: fn["parameters"] for fn in self.function_data
        }


class CreateTries:
    """Builds token-level execution constraints for function names using a
    prefix tree.

    Attributes:
        function_data: A list of dictionaries containing raw function
            definitions.
        function_names: A list of extracted function identifier strings.
        FnTrie: A TokenTrie instance representing allowed sequence branches.
        model: The small language model instance used to encode string tokens.
    """

    def __init__(self, function_data: list[dict[str, Any]],
                 model: Small_LLM_Model) -> None:
        """Initializes CreateTries, extracts function names, and populates
        the Trie structure.

        Args:
            function_data: Raw specification structures for available
                functions.
            model: The language model instance used to tokenize identifiers.
        """
        self.function_data = function_data
        self.function_names = self.get_fn_names()
        self.FnTrie = TokenTrie()
        self.model = model
        self.insert_fns()

    def make_encode(self, func: str) -> list[int]:
        """Converts a function name string into its model token IDs.

        Args:
            func: The name string of the target function.

        Returns:
            A list of vocabulary token integers representing the string.
        """
        encode = self.model.encode(func)[0].tolist()
        return cast(list[int], encode)

    def insert_fns(self) -> None:
        """Encodes all extracted function names and adds them to the
        validation tree."""
        for func in self.function_names:
            self.FnTrie.add_token_branch(
                        self.make_encode(func)
                        )

    def get_fn_names(self) -> list[str]:
        """Extracts text identifiers for all functions present in the input.

        Returns:
            A list of function name strings.
        """
        return [fn["name"] for fn in self.function_data]


class BuildTokenCache:
    """Categorizes the model vocabulary into sets based on string syntax
    matches.

    Attributes:
        str_to_tok: The source vocabulary mapping vocabulary strings to
            token IDs.
        token_cache: A dictionary tracking pre-filtered token ID sets
            indexed by syntactic category names.
    """

    def __init__(self, str_to_tok: dict[str, int]) -> None:
        """Initializes BuildTokenCache with the model vocabulary mapping.

        Args:
            str_to_tok: Complete vocabulary map where keys are string
                representations and values are token IDs.
        """
        self.str_to_tok = str_to_tok
        self.token_cache: dict[str, set[int]] = {}

    def filter_tokens(self, f: Callable[[str], bool]) -> set[int]:
        """Filters vocabulary tokens matching a specific boolean predicate
        function.

        Args:
            f: A predicate function evaluating a cleaned token string and
                returning a boolean.

        Returns:
            A set of token integer IDs that satisfied the condition.
        """
        return {
                v for k, v in self.str_to_tok.items() if f(k.replace("Ġ", " "))
            }

    def build_cache(self) -> None:
        """Populates the cache dictionary with pre-filtered sets for JSON
        generation control."""
        self.token_cache["START_PARAM"] = self.filter_tokens(
            lambda s: s.strip().startswith('{')
        )

        self.token_cache["ANY_VALUE_START"] = self.filter_tokens(
            lambda s: any(s.strip().startswith(c) for c in ['"', '['])
        )
        self.token_cache["BOOL_CONTINUE"] = self.filter_tokens(
            lambda s: any(s.strip().startswith(c) for c in [
                "e", "ue", "rue", "true",
                "se", "lse", "alse", "false",
                "l", "ll", "ull", "null"
            ])
        )

        self.token_cache["ARR_START_END"] = self.filter_tokens(
            lambda s: s.strip().startswith('["') or s.strip().endswith('"]')
            or s.strip() in ('[', ']')
        )

        self.token_cache["NUMBER_CONTINUE"] = self.filter_tokens(
            lambda s: bool(re.match(
                r'^-?[0-9]*\.?[0-9]*([eE][+-]?[0-9]*)?$', s.strip()))
            and s.strip() != ""
        )

        self.token_cache["STRING_INNER"] = self.filter_tokens(
            lambda s: not any(ord(c) < 32 for c in s)
        )

        self.token_cache["NEXT_DELIMITER"] = self.filter_tokens(
            lambda s: any(c in s for c in [',', '}', ']']) or
            any(s.strip().startswith(c) for c in [',', '}', ']'])
        )

        self.token_cache["EXPECT_COLON"] = self.filter_tokens(
            lambda s: ':' in s or s.strip().startswith(':')
        )
