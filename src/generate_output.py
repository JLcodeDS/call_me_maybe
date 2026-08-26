from llm_sdk import Small_LLM_Model
import numpy as np
from typing import Any
from datetime import datetime


class GenerateOutput:
    def __init__(
            self, prompts: dict[str, str], constraints: ConstrainModel
            ) -> None:
        self.model = Small_LLM_Model()
        self.prompts = prompts
        self.constraints = constraints
        self.funcs = constraints.function_data

    def get_answer_to_prompt(self, prompt: str) -> str:
        wrapped_prompt = (f"Function definitions: {self.funcs}." +
                          f"Response output format: {
                              self.constraints.output_schema}" +
                          "Use the function definitions to answer" +
                          "the following prompt in the response output" +
                          f" format and nothing more:\n{prompt}\n")
        response_id = len(wrapped_prompt)
        wrapped_prompt += '{\n"prompt":"' + f"{prompt}" + '",\n"name":'
        encode = self.model.encode(wrapped_prompt)[0].tolist()
        max_logit_id = None
        print(self.model._tokenizer.eos_token_id)
        token_count = 0
        start = datetime.now()
        while (token_count < 20):
            logits = self.model.get_logits_from_input_ids(encode)
            max_logit_id = np.argmax(logits).item()
            encode.append(max_logit_id)
            token_count += 1
        end = datetime.now()
        print(f"Time elapsed: {end - start}")
        return self.model.decode(encode)[response_id:]

    def run_all_prompts(self) -> list[str]:
        outputs: list[str] = []
        for prompt in self.prompts:
            outputs.append(self.get_answer_to_prompt(prompt))
        return outputs


class ConstrainModel:
    def __init__(self, function_data: list[
            dict[str, Any]], output_schema: dict[str, Any]) -> None:
        self.function_data = function_data
        self.output_schema = output_schema

    def get_function_names(self) -> list[str]:
        return [func["name"] for func in self.function_data]
