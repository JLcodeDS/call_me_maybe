from pydantic import BaseModel, ValidationError, Field, create_model
from typing import Literal, Union
from enum import Enum
import json
from sys import argv
from data_models import FunctionCall, FunctionDefinitions


class LoadData:
    def _read_cli_params(self) -> None:
        dict_params = {
            "functions_definition": "data/input/functions_definition.json",
            "input": "data/input/function_calling_tests.json",
            "output": "data/output/function_calls.json"
        }

        usage_err_str = (
            "Usage: uv run python -m src"
            "[--functions_definition <function_definition_file>]"
            "[--input <input_file>] [--output <output_file>]"
                        )

        allowed_args = ["--functions_definition", "--input", "--output"]

        if len(argv) == 1:
            self.dict_params = dict_params
        elif len(argv) > 7:
            raise Exception(
                "ERROR: too many arguments.\n" + usage_err_str
            )
        else:
            params = argv[1:]
            param_pairs = list(zip(params[::2], params[1::2]))
            if len(params) % 2 or not len(param_pairs):
                raise Exception(
                    "ERROR: missing values for specified arguments.\n" +
                    usage_err_str
                                )
            for pair in param_pairs:
                if pair[0] not in allowed_args:
                    raise Exception(
                        "ERROR: duplicate or incorrect arguments.\n" +
                        usage_err_str
                    )
                dict_params[pair[0].replace("--", "")] = pair[1]
                allowed_args.remove(pair[0])
            self.dict_params = dict_params

    def _load_json(self) -> None:
        try:
            with open(self.dict_params["functions_definition"]) as funcs:
                self.function_data = json.load(funcs)
            with open(self.dict_params["input"]) as prompts:
                self.prompt_data = json.load(prompts)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error when parsing json data: {e}")

    def _validate_data(self) -> None:
        errors_lst: list = []
        for prompt in self.prompt_data:
            try:
                FunctionCall.model_validate(prompt)
            except ValidationError as e:
                for error in e.errors():
                    loc = " -> ".join(str(x) for x in error["loc"])
                    errors_lst.append(f"FunctionCall [{loc}]: {error['msg']}")
        for functions in self.function_data:
            try:
                FunctionDefinitions.model_validate(functions)
            except ValidationError as e:
                for error in e.errors():
                    loc = " -> ".join(str(x) for x in error["loc"])
                    errors_lst.append(
                        f"FunctionDefinitions [{loc}]: {error['msg']}")
        if len(errors_lst):
            raise Exception(
                "ERROR: Invalid data. Found following errors:\n" +
                "\n".join(errors_lst)
            )

    def load_data(self) -> None:
        self._read_cli_params()
        self._load_json()
        self._validate_data()


