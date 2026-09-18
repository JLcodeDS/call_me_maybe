"""Model inference and constrained generation execution pipeline.

This module orchestrates the generation of function-calling outputs by
leveraging a TokenTrie for constrained function name detection and a
PydanticFSM for grammar-constrained JSON parameter extraction.
"""

from llm_sdk import Small_LLM_Model  # type:ignore
import numpy as np
from typing import Any, cast
from .data_models import TokenTrie, DynamicParameterModel
from .data_preparation import PrepareFnData
from .fsms import PydanticFSM
from pydantic import ValidationError
import json


class RunModel:
    """Manages constrained generation steps to produce structured function
    calls from an LLM.

    This class handles the end-to-end execution of a structured call request.
    It escapes the prompt, uses a Trie to generate valid function names, and
    applies a finite state machine (FSM) to restrict token generation until
    a valid parameter JSON structure is achieved.

    Attributes:
        model: The small language model instance used for inference.
        trie: A prefix tree constraint containing allowed token paths for
            function names.
        prompt: The processed user input prompt string with escaped characters.
        clean_fn_data: Extracted and structured definitions of valid functions.
        token_cache: Categorized dictionary tracking valid token IDs by JSON
            patterns.
        dynamic_models: Factory class holding the compiled Pydantic
            validation structures.
        parameters: A dictionary capturing extracted arguments
            (initialized empty).
        function_name: The determined function string resolved during
            inference.
        encoded_response: A running list of token IDs representing the
            complete generated track.
        response_id: The structural boundary marker offset tracking
            string positions.
    """

    def __init__(self, prompt: str, model: Small_LLM_Model,
                 trie: TokenTrie,
                 clean_fn_data: PrepareFnData,
                 dynamic_params: DynamicParameterModel,
                 token_cache: dict[str, set[int]]
                 ):
        """Initializes RunModel and prepares the query prompt text.

        Args:
            prompt: Raw string prompt input submitted by the user.
            model: The language model wrapper object.
            trie: Prefix execution token structures for valid identifiers.
            clean_fn_data: Pre-processed metadata definitions.
            dynamic_params: Active factory instance holding target schemas.
            token_cache: Categorized sets of integer vocabulary tokens.
        """
        self.model = model
        self.trie = trie
        self.prompt = prompt.replace("\\", r"\\").replace(r'"', '\\"')
        print(self.prompt)
        self.clean_fn_data = clean_fn_data
        self.token_cache = token_cache
        self.dynamic_models = dynamic_params
        self.parameters: dict[str, Any] = {}

    def run(self) -> dict[str, Any] | Any:
        """Executes the sequential workflow to isolate a function name and
        fill its parameters.

        Finds the target function identifier, populates required argument
        tokens via FSM mapping, and extracts the result. Finally, validates
        against Pydantic definitions before export.

        Returns:
            A validated model dictionary if parsing succeeds, or a raw decoded
            JSON dictionary representation if structural validation fails.
        """
        print("Finding function...", flush=True)
        self.find_fn_name()
        print("\nFinding parameters...", flush=True)
        self.find_params()
        json_str = self.model.decode(self.encoded_response)
        output = json_str[len(self.get_wrapped_prompt()):]
        try:
            valid = self.dynamic_models.fn_models[
                self.function_name].model_validate_json(output)
            return valid.model_dump()
        except ValidationError:
            return json.loads(output)

    def find_fn_name(self) -> None:
        """Extracts an allowed function name token sequence through
        constrained logit masking."""
        fn_prompt = self.get_wrapped_prompt()
        self.response_id = len(fn_prompt)
        fn_prompt += ('{"prompt": "' + self.prompt +
                      '", "name": "')
        encoded_prompt = self.model.encode(fn_prompt)[0].tolist()
        encoded_fn: list[int] = []
        while True:
            logits = self.model.get_logits_from_input_ids(encoded_prompt)
            allowed_next_tokens = self.trie.get_next_allowed_tokens(
                encoded_fn)
            if allowed_next_tokens is None:
                break
            max_logit_id = self.mask_logits(logits, list(allowed_next_tokens))
            encoded_prompt.append(max_logit_id)
            encoded_fn.append(max_logit_id)
            print(f"{self.model.decode(max_logit_id)}", end="", flush=True)
        self.encoded_response = encoded_prompt
        self.function_name = self.model.decode(encoded_fn)

    def find_params(self) -> None:
        """Populates the function parameters matching the structural schema
        via FSM constraints."""
        params = self.dynamic_models.fn_models[self.function_name]
        fsm = PydanticFSM(params, self.token_cache)
        self.encoded_response.extend(self.get_param_prompt_tokens())
        entire_response_buffer = self.model.decode(self.encoded_response)[
            len(self.get_wrapped_prompt()):]
        response_buffer = ""
        while True:
            allowed_next_tokens = fsm.get_allowed_tokens(
                response_buffer, entire_response_buffer)
            if allowed_next_tokens is None:
                break
            logits = self.model.get_logits_from_input_ids(
                self.encoded_response)
            max_logit_id = self.mask_logits(logits, list(
                                allowed_next_tokens))
            self.encoded_response.append(max_logit_id)
            new_token = self.model.decode(max_logit_id)
            print(f"{new_token}", end="", flush=True)
            response_buffer += new_token
            entire_response_buffer += new_token

    def get_wrapped_prompt(self) -> str:
        """Constructs the system orchestration prompt block and syntax
        instructions.

        Returns:
            The complete context window instruction string including active
            function schemas.
        """
        return ("You are a function-calling assistant.\n"
                f"Available functions:\n{self.clean_fn_data.function_data}\n" +
                "Output the call in JSON format matching the schema:\n" +
                '{"prompt": <prompt>, "name": "<function_name>",' +
                ' "parameters": {<param_name>: <param_value>, ...}}' +
                "Rule: For the regex field, NEVER output literal matches. "
                "Always use proper regex sets "
                "(e.g. '[aeiouAEIOU]', '[0-9]+', '\\bword\\b'). "
                "For replacement, if asked for a character (e.g. asterisks), "
                "output EXACTLY ONE character (e.g. '*')."
                f"\nUser request: {self.prompt}\n"
                '\nJSON output:\n')

    def get_param_prompt_tokens(self) -> list[int]:
        """Encodes the literal json key connector boundary string into model
        tokens.

        Returns:
            A list of token integer values representing the transition string.
        """
        encode = self.model.encode('", "parameters": ')[0].tolist()
        return cast(list[int], encode)

    @staticmethod
    def mask_logits(logits: list[float], allowed_tokens: list[int]
                    ) -> int:
        """Applies negative infinity masking over unauthorized vocabulary
        logit scores.

        Args:
            logits: Complete float logit score vector from the model output.
            allowed_tokens: Intersected list of valid vocabulary indices.

        Returns:
            The integer index corresponding to the allowed token with the
            maximum score.
        """
        logits_arr = np.array(logits)
        masked = np.full_like(logits_arr, float("-inf"))
        masked[allowed_tokens] = logits_arr[allowed_tokens]
        return int(np.argmax(masked))
