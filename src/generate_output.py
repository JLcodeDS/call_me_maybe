from llm_sdk import Small_LLM_Model
import numpy as np
from typing import Any
from datetime import datetime
import json


class ConstrainModel:
    def __init__(self, function_data: list[
            dict[str, Any]], output_schema: dict[str, Any]) -> None:
        self.output_schema = output_schema
        self.function_data = function_data


class ModelInterface:
    def __init__(
            self, constraints: ConstrainModel
            ) -> None:
        self.model = Small_LLM_Model()
        self.constraints = constraints
        self.funcs = constraints.function_data
        self.function_names = self.get_function_names()

    def get_wrapped_prompt(self, raw_prompt: str) -> tuple[int, str]:
        wrapped_prompt = (f"Function definitions: {self.funcs}." +
                          f"Response output format: {
                              self.constraints.output_schema}" +
                          "Use the function definitions to answer" +
                          "the following prompt in the response output" +
                          f" format and nothing more:\n{raw_prompt}\n")
        response_start_id = len(wrapped_prompt)
        wrapped_prompt += '{\n"prompt":"' + f"{raw_prompt}" + '",\n"name": "'
        return (response_start_id, wrapped_prompt)

    def run_constrained_model(self, raw_prompt: str):
        response_start, wrapped_prompt = self.get_wrapped_prompt(raw_prompt)
        encode = self.model.encode(wrapped_prompt)[0].tolist()
        token_count = 0
        start = datetime.now()
        fn_name_id = len(wrapped_prompt)
        function_token_ids = [
            tok for name in self.get_function_name_token_ids() for tok in name]
        while (wrapped_prompt[fn_name_id:] not in self.function_names):
            
            logits = self.model.get_logits_from_input_ids(encode)
            max_logit_id = np.argmax(
                self.mask_logits(logits, function_token_ids)).item()
            encode.append(max_logit_id)
            token_count += 1
        end = datetime.now()
        print(f"Time elapsed: {end - start}")
        return self.model.decode(encode)[response_start:]

    def get_clean_vocab_tokens(self) -> dict[int, str]:
        vocab_file = self.model.get_path_to_vocab_file()
        with open(vocab_file) as vocab:
            vocab_dict = json.loads(vocab.read())
        return {
            k: v for k, v in vocab_dict.items() if k.isprintable()
        }

    def init_fn_name_tree(self, fn_tokens: list[list[int]]) -> None:
        ## Fix tree first

    def get_function_name_token_ids(self) -> list[list[int]]:
        function_token_ids = [
            self.model.encode(func["name"])[0].tolist() for func in self.funcs]
        return function_token_ids

    def get_function_names(self) -> list[str]:
        return [func["name"] for func in self.funcs]

    def get_allowed_tokens(self, last_token: int, fn_name_tree: dict[Any]) -> list[int]:

    @staticmethod
    def mask_logits(logits: list[float], allowed_tokens: list[int]
                    ) -> list[float]:
        masked_set = ~np.isin(np.arange(0, len(logits)), allowed_tokens)
        logits_np = np.array(logits)
        logits_np[masked_set] = float("-inf")
        return list(logits_np)
