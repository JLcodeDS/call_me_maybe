from llm_sdk import Small_LLM_Model
import numpy as np


class GenerateOutput:
    def __init__(self, prompt: str) -> None:
        self.model = Small_LLM_Model()
        self.prompt = prompt

    def run_model(self, funcs: str) -> None:
        wrapped_prompt = f"I have the following functions: {funcs}. Give the function call as a json output to the following prompt and nothing more:\n " + self.prompt
        encode = self.model.encode(wrapped_prompt)[0].tolist()
        max_logit_id = None
        print(self.model._tokenizer.eos_token_id)
        token_count = 0
        while (max_logit_id != self.model._tokenizer.eos_token_id and token_count < 100):
            logits = self.model.get_logits_from_input_ids(encode)
            max_logit_id = np.argmax(logits).item()
            # max_ids = np.argsort(logits, descending=True)[:3]
            # print(max_ids)
            # print("3 max ", self.model.decode(max_ids))
            # print(logits[max_ids])
            encode.append(max_logit_id)
            token_count += 1
        print(self.model.decode(encode))

    def softmax(self, raw_logits: list[float]) -> list[float]:
        pass
