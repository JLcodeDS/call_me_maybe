from pydantic import BaseModel, Field, create_model, RootModel
from enum import Enum
from typing import Union, Literal


class AllowedTypes(BaseModel):
    type: str = Field(min_length=3, max_length=100)


class FunctionDefinitions(BaseModel):
    name: str = Field(max_length=100, min_length=3)
    description: str = Field(max_length=500)
    parameters: dict[str, AllowedTypes]
    returns: AllowedTypes


class FunctionCall(BaseModel):
    prompt: str = Field(min_length=5, max_length=500)


class CreateResponseModel:
    def __init__(self, function_data: dict[str, str | dict[str, str | dict[str, str]]]):
        self.function_data = function_data
        self.ResponseModel = None
    
    #def create_response_model(self) -> None:
    #    function_models = 
    #    ModelResponse = create_model(
    #        "ModelResponse",
            
    #    )

    def extract_param_dict(self, func_dict):
        param_type_conversion = {
            "string": str,
            "number": float,
            "boolean": bool
        }
        params = {}
        for param_name, param_info in func_dict[
                    "parameters"].items():
            params[param_name] = (param_type_conversion[
                param_info["type"]], ...)
        return params

    def create_schema(self) -> type[RootModel]:
        function_models = []
        for func in self.function_data:
            func_name = func["name"]
            ParamsModel = create_model(
                f"{func_name}_params",
                **self.extract_param_dict(func)
            )
            FunctionModel = create_model(
                f"{func_name}_model",
                prompt = (str, Field(max_length=500, min_length=3)),
                name = (Literal[func_name], func_name),
                parameters = (ParamsModel, ...)
            )
            function_models.append(FunctionModel)
        function_union = Union[tuple(function_models)]
        
        ModelResponse = create_model(
            "ModelResponse",
            __base__ = RootModel[function_union]
        )
        return ModelResponse