from pydantic import ValidationError
import json
from .data_models import FunctionDefinitions, PromptSchema
from argparse import ArgumentParser


class LoadData:
    def _read_cli_params(self) -> None:
        parser = ArgumentParser(
            description=(
                "Process function definition, input, and output files."),
            usage=("uv run python -m src [--functions_definition <file>]" +
                   "[--input <file>] [--output <file>]")
        )
        parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json",
            help="Path to the functions definition JSON file"
        )
        parser.add_argument(
            "--input",
            default="data/input/function_calling_tests.json",
            help="Path to the input JSON file"
        )
        parser.add_argument(
            "--output",
            default="data/output/function_calling_results.json",
            help="Path to the output JSON file"
        )
        self.dict_params = vars(parser.parse_args())

    def _load_json(self) -> None:
        try:
            with open(self.dict_params["functions_definition"]) as funcs:
                self.function_data = json.load(funcs)
        except Exception as e:
            raise Exception("Error when loading file " +
                            f"{self.dict_params['input']}: {e}")
        try:
            with open(self.dict_params["input"]) as prompts:
                self.prompt_data = json.load(prompts)
        except Exception as e:
            raise Exception("Error when loading file " +
                            f"{self.dict_params['input']}: {e}")

    def _validate_data(self) -> None:
        errors_lst: list[str] = []
        for prompt in self.prompt_data:
            try:
                PromptSchema.model_validate(prompt)
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
