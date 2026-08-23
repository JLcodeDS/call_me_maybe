from llm_sdk import Small_LLM_Model
import numpy as np


class GenerateOutput:
    def __init__(self, prompt: str) -> None:
        self.model = Small_LLM_Model()
        
        self.prompt = "Answer the following prompt: " + prompt

    def run_model(self) -> None:
        encode = self.model.encode(self.prompt)[0].tolist()
        max_logit_idx = None
        print(self.model._tokenizer.eos_token_id)
        token_count = 0
        while (max_logit_idx != self.model._tokenizer.eos_token_id and token_count < 50):
            logits = self.model.get_logits_from_input_ids(encode)
            max_logit_idx = np.argmax(logits).item()
            encode.append(max_logit_idx)
            token_count += 1
            print(self.model.decode(encode))
