from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
from typing import Any
from .data_models import TokenTrie
from .data_preparation import PrepareFnData
from .fsms import FSMDispatch


class RunModel:
    def __init__(self, prompt: str, model: Small_LLM_Model,
                 trie: TokenTrie, output_schema: dict[str, Any],
                 clean_fn_data: PrepareFnData, token_cache: dict[str, set[int]]
                 ):
        self.model = model
        self.trie = trie
        self.prompt = prompt
        self.output_schema = output_schema
        self.clean_fn_data = clean_fn_data
        self.token_cache = token_cache
        self.parameters: dict[str, Any] = {}

    def run(self) -> str:
        self.find_fn_name()
        print(f"function: {self.function_name}\nsearching params...")
        self.find_params()
        json_str = self.model.decode(self.encoded_response)[:-2] + "}}"
        return json_str[len(self.get_wrapped_prompt()):]

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
            max_logit_id = np.argmax(
                self.mask_logits(logits, list(allowed_next_tokens))).item()
            encoded_prompt.append(max_logit_id)
            encoded_fn.append(max_logit_id)
        self.encoded_response = encoded_prompt
        self.function_name = self.model.decode(encoded_fn)

    def find_params(self) -> None:
        params = self.clean_fn_data.fn_params[self.function_name]
        self.encoded_response.extend(self.get_param_prompt_tokens())
        for i, (p, p_type) in enumerate(params.items()):
            encoded_param: list[int] = []
            param_extension = f'"{p}": '
            self.encoded_response.extend(
                self.model.encode(param_extension)[0].tolist())
            fsm_dispatch = FSMDispatch(p_type, self.token_cache, self.model)
            FsmType = fsm_dispatch.dispatch()
            while not FsmType.is_end_state(encoded_param):
                print(self.model.decode(self.encoded_response))
                logits = self.model.get_logits_from_input_ids(
                    self.encoded_response)
                allowed_next_tokens = FsmType.get_allowed_next_tokens(
                    encoded_param)
                assert allowed_next_tokens is not None
                max_logit_id = np.argmax(
                                self.mask_logits(logits, list(
                                    allowed_next_tokens))).item()
                self.encoded_response.append(max_logit_id)
                encoded_param.append(max_logit_id)
            if i < len(params):
                if p_type not in ("number", "integer"):
                    self.encoded_response.extend(
                                self.model.encode(", ")[0].tolist())
                else:
                    self.encoded_response.extend(
                                    self.model.encode(" ")[0].tolist())

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
        return self.model.encode('", "parameters": {')[0].tolist()

    @staticmethod
    def mask_logits(logits: list[float], allowed_tokens: list[int]
                    ) -> list[float]:
        masked = [float("-inf")] * len(logits)
        for token_id in allowed_tokens:
            masked[token_id] = logits[token_id]
        return masked
