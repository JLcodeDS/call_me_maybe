from pydantic import BaseModel, Field
from typing import Any


class AllowedTypes(BaseModel):
    type: str = Field(min_length=3, max_length=100)


class FunctionDefinitions(BaseModel):
    name: str = Field(max_length=100, min_length=3)
    description: str = Field(max_length=500)
    parameters: dict[str, AllowedTypes]
    returns: AllowedTypes


class JSONOutput(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    name: str = Field(min_length=3, max_length=500)
    parameters: dict[str, Any]


class PromptSchema(BaseModel):
    prompt: str = Field(min_length=5, max_length=500)


class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_word = False


class TokenTrie:
    def __init__(self):
        self.first_node = TrieNode()

    def add_token_branch(self, tokens: list[int]) -> None:
        current_node = self.first_node
        for t in tokens:
            if t not in self.first_node.children:
                current_node.children[t] = TrieNode()
            current_node = current_node.children[t]
        current_node.end_word = True

    def get_next_allowed_tokens(self, tokens: list[int]) -> set[int] | None:
        current_node = self.first_node
        if not len(tokens):
            return current_node.children.keys()
        for t in tokens:
            current_node = current_node.children[t]
        if current_node.end_word:
            return None
        else:
            return current_node.children.keys()


"""
Make dict of pydantic models to index after having found the
correct function name !!
index on the function name and then model is free flowing
"""
# class CreateResponseModel:
#     def __init__(self, function_data: list[dict[str, Any]]):
#         self.function_data = function_data
#         self.ResponseModel = None

#     def extract_param_dict(self, func_dict: dict[str, Any]
#                            ) -> dict[str, type]:
#         param_type_conversion = {
#             "string": str,
#             "number": float,
#             "boolean": bool
#         }
#         params: dict[str, Any] = {}
#         for param_name, param_info in func_dict[
#                     "parameters"].items():
#             params[param_name] = (param_type_conversion[
#                 param_info["type"]])
#         return params

#     def create_schema(self) -> type[RootModel]:
#         function_models: list[type[BaseModel]] = []
#         for func in self.function_data:
#             func_name: str = func["name"]
#             param_fields: dict[str, Any] = self.extract_param_dict(func)
#             ParamsModel = create_model(
#                 f"{func_name}_params",
#                 **param_fields
#             )
#             FunctionModel = create_model(
#                 f"{func_name}_model",
#                 prompt = (str, Field(max_length=500, min_length=3)),
#                 name = (Literal[func_name], func_name), # type: ignore[valid-type]
#                 parameters = (ParamsModel, ...)
#             )
#             function_models.append(FunctionModel)

#         function_union = Union[tuple(function_models)]  # type: ignore[valid-type]

#         ModelResponse = create_model(
#             "ModelResponse",
#             __base__ = RootModel[function_union]
#         )
#         return ModelResponse
