"""CLI parameter parsing, JSON loading, and Pydantic schema validation.

This module provides data ingest components to parse command line parameters,
load target evaluation files, and rigorously validate function structural
definitions against strict Pydantic schemas.
"""

from pydantic import ValidationError
import json
from .data_models import FunctionDefinitions, PromptSchema
from argparse import ArgumentParser


class LoadData:
    """Handles command-line arguments, JSON ingestion, and structural data
    validation.

    This class parses input and output file paths from the CLI arguments, reads
    their JSON contents into memory, and performs structural verification using
    Pydantic schema definitions before runtime processing starts.

    Attributes:
        dict_params: A dictionary tracking runtime file system paths extracted
            from argument parameters.
        function_data: A list of loaded raw dictionary configurations for
            valid functions.
        prompt_data: A list of loaded raw dictionary items representing
            generation prompts.
    """

    def _read_cli_params(self) -> None:
        """Parses execution options and target file configurations from
        command line arguments."""
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
        """Reads configuration datasets into local properties from filesystem
        paths.

        Raises:
            Exception: If file access or JSON structure decoding fails on the
                provided parameter paths.
        """
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
        """Enforces schema matching across function configurations and prompt
        structures.

        Accumulates structural type mismatches and missing fields via Pydantic
        model validations, formatting errors into a unified execution summary
        report.

        Raises:
            Exception: Contains a formatted list of all discovered schema
                errors if any loaded entity fails validation.
        """
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
        """Executes sequential data loading pipeline stages.

        Orchestrates CLI collection, JSON ingestion, and structural validation
        to ensure data consistency before continuing model setup.
        """
        self._read_cli_params()
        self._load_json()
        self._validate_data()
