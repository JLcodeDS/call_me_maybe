"""Data models and prefix tree (Trie) structures for LLM parameter validation.

This module defines Pydantic schemas for functions and prompts, implements
a Trie data structure for token-level constraints, and provides a builder
class to dynamically generate Pydantic models from runtime JSON schemas.
"""

from pydantic import BaseModel, Field, create_model as pd_create_model
from json_schema_to_pydantic import create_model  # type:ignore
from typing import Any
from collections.abc import KeysView


class AllowedTypes(BaseModel):
    """Pydantic model representing a validation schema for data types.

    Attributes:
        type: The string representation of the allowed data type.
    """
    type: str = Field(min_length=3, max_length=100)


class FunctionDefinitions(BaseModel):
    """Pydantic model that defines the metadata schema for functions.

    Attributes:
        name: The target identifier of the function.
        description: A text explanation detailing what the function does.
        parameters: A mapping of parameter names to their AllowedTypes schema.
        returns: The expected schema configuration for the function's return
        value.
    """
    name: str = Field(max_length=500, min_length=3)
    description: str = Field(max_length=1000)
    parameters: dict[str, AllowedTypes]
    returns: AllowedTypes


class PromptSchema(BaseModel):
    """Pydantic model representing a single evaluation prompt.

    Attributes:
        prompt: The input text prompt to be submitted to the model.
    """
    prompt: str = Field(min_length=3, max_length=1000)


class TrieNode:
    """A single node within the TokenTrie prefix tree structure.

    Attributes:
        children: A mapping connecting vocabulary token IDs to subsequent
            TrieNode instances.
        end_word: A boolean flag indicating whether the current node marks
            the end of a complete token sequence.
    """
    def __init__(self) -> None:
        """Initializes an empty TrieNode instance."""
        self.children: dict[int, TrieNode] = {}
        self.end_word = False


class TokenTrie:
    """Prefix tree structure optimized for managing valid vocabulary token
    paths.

    Attributes:
        first_node: The entry point or root TrieNode of the tree structure.
    """
    def __init__(self) -> None:
        """Initializes a TokenTrie instance with a root node."""
        self.first_node = TrieNode()

    def add_token_branch(self, tokens: list[int]) -> None:
        """Inserts a sequence of token IDs into the prefix tree structure.

        Args:
            tokens: A sequential list of token identifiers representing
                a valid execution branch.
        """
        current_node = self.first_node
        for t in tokens:
            if t not in self.first_node.children:
                current_node.children[t] = TrieNode()
            current_node = current_node.children[t]
        current_node.end_word = True

    def get_next_allowed_tokens(self, tokens: list[int]
                                ) -> KeysView[int] | None:
        """Retrieves valid token keys that can follow the provided sequence.

        Args:
            tokens: A list of prefix tokens representing the current generation
                state.

        Returns:
            A view collection containing valid subsequent token keys if found,
            or None if the prefix reaches a finalized terminal sequence.
        """
        current_node = self.first_node
        if not len(tokens):
            return current_node.children.keys()
        for t in tokens:
            current_node = current_node.children[t]
        if current_node.end_word:
            return None
        else:
            return current_node.children.keys()


class DynamicParameterModel:
    """Factory builder that dynamically converts JSON schemas into Pydantic
    models.

    Attributes:
        function_params: A schema map containing dynamic parameter definitions
            grouped by function name.
        fn_models: A map linking function names to their generated full
            Pydantic wrapper models.
        param_models: A map linking function names to their internal
            parameters-only Pydantic validation models.
    """
    def __init__(self, function_params: dict[str, dict[str, Any]]):
        """Initializes DynamicParameterModel and starts model compilation.

        Args:
            function_params: Raw schema configurations indexed by
            function name.
        """
        self.function_params = function_params
        self.fn_models, self.param_models = self.create_output_models()

    def create_output_models(self) -> tuple[dict[str, type[BaseModel]],
                                            dict[str, type[BaseModel]]]:
        """Dynamically creates validation classes for both parameters and
        full structures.

        Returns:
            A tuple containing two elements:
            - A dictionary mapping function names to combined evaluation
                models.
            - A dictionary mapping function names to dedicated parameter
                validation models.
        """
        fn_models: dict[str, type[BaseModel]] = {}
        param_models: dict[str, type[BaseModel]] = {}
        for fn_name, params in self.function_params.items():
            params = {
                "title": "parameters",
                "type": "object",
                "properties": params
            }
            param_model = create_model(params)
            pd_model = pd_create_model(
                fn_name, prompt=str, name=str, parameters=param_model)
            fn_models[fn_name] = pd_model
            param_models[fn_name] = param_model
        return fn_models, param_models
