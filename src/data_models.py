from pydantic import BaseModel, Field, create_model as pd_create_model
from json_schema_to_pydantic import create_model  # type:ignore
from typing import Any
from collections.abc import KeysView


class AllowedTypes(BaseModel):
    type: str = Field(min_length=3, max_length=100)


class FunctionDefinitions(BaseModel):
    name: str = Field(max_length=500, min_length=3)
    description: str = Field(max_length=1000)
    parameters: dict[str, AllowedTypes]
    returns: AllowedTypes


class PromptSchema(BaseModel):
    prompt: str = Field(min_length=3, max_length=1000)


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.end_word = False


class TokenTrie:
    def __init__(self) -> None:
        self.first_node = TrieNode()

    def add_token_branch(self, tokens: list[int]) -> None:
        current_node = self.first_node
        for t in tokens:
            if t not in self.first_node.children:
                current_node.children[t] = TrieNode()
            current_node = current_node.children[t]
        current_node.end_word = True

    def get_next_allowed_tokens(self, tokens: list[int]) -> KeysView[
            int] | None:
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
    def __init__(self, function_params: dict[str, dict[str, Any]]):
        self.function_params = function_params
        self.fn_models, self.param_models = self.create_output_models()

    def create_output_models(self) -> tuple[dict[str, type[BaseModel]],
                                            dict[str, type[BaseModel]]]:
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
