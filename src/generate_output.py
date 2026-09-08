from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
from typing import Any
from .data_models import TokenTrie, DynamicParameterModel
from .data_preparation import PrepareFnData
from .fsms import PydanticFSM
from pydantic import ValidationError, BaseModel
from datetime import datetime


class RunModel:
    def __init__(self, prompt: str, model: Small_LLM_Model,
                 trie: TokenTrie,
                 clean_fn_data: PrepareFnData,
                 dynamic_params: DynamicParameterModel,
                 token_cache: dict[str, set[int]]
                 ):
        self.model = model
        self.trie = trie
        self.prompt = prompt.replace('\\"', '"').replace('"', '\\"')
        self.clean_fn_data = clean_fn_data
        self.token_cache = token_cache
        self.dynamic_models = dynamic_params
        self.parameters: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        self.find_fn_name()
        self.find_params()
        json_str = self.model.decode(self.encoded_response)
        output = json_str[len(self.get_wrapped_prompt()):]
        try:
            valid = self.dynamic_models.fn_models[
                self.function_name].model_validate_json(output)
            return valid.model_dump()
        except ValidationError as e:
            print(f"Error for \n{output}")
            raise e

    def find_fn_name(self) -> None:
        fn_prompt = self.get_wrapped_prompt()
        self.response_id = len(fn_prompt)
        fn_prompt += '{"prompt": "' + self.prompt + '", "name": "'
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
        self.encoded_response = encoded_prompt
        self.function_name = self.model.decode(encoded_fn)

    def find_params(self) -> None:
        params = self.dynamic_models.fn_models[self.function_name]
        fsm = PydanticFSM(params, self.token_cache)
        self.encoded_response.extend(self.get_param_prompt_tokens())
        response_buffer = ""
        while True:
            allowed_next_tokens = fsm.get_allowed_tokens(response_buffer)
            if allowed_next_tokens is None:
                break
            logits = self.model.get_logits_from_input_ids(
                self.encoded_response)
            max_logit_id = self.mask_logits(logits, list(
                                allowed_next_tokens))
            self.encoded_response.append(max_logit_id)
            response_buffer += self.model.decode(max_logit_id)

    def get_wrapped_prompt(self) -> str:
        return ("You are a function-calling assistant.\n"
                f"Available functions:\n{self.clean_fn_data.function_data}\n" +
                "Output the call in JSON format matching the schema:\n" +
                '{"prompt": <prompt>, "name": "<function_name>",' +
                ' "parameters": {<param_name>: <param_value>, ...}}\n' +
                "Rule: For the regex field, NEVER output literal matches. "
                "Always use proper regex sets "
                "(e.g. '[aeiouAEIOU]', '[0-9]+', '\\bword\\b'). "
                "For replacement, if asked for a character (e.g. asterisks), "
                "output EXACTLY ONE character (e.g. '*')."
                f"\nUser request: {self.prompt}\n"
                '\nJSON output:\n')

    def get_param_prompt_tokens(self) -> list[int]:
        return self.model.encode('", "parameters": ')[0].tolist()

    @staticmethod
    def mask_logits(logits: list[float], allowed_tokens: list[int]
                    ) -> int:
        logits_arr = np.array(logits)
        masked = np.full_like(logits_arr, float("-inf"))
        masked[allowed_tokens] = logits_arr[allowed_tokens]
        return int(np.argmax(masked))
