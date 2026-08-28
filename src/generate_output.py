from llm_sdk import Small_LLM_Model
import numpy as np
from typing import Any
from datetime import datetime
from .data_models import TokenTrie


class CleanInput:
    def __init__(self, function_data: dict[str, Any],
                 prompt_data: dict[str, str]) -> None:
        self.function_data = function_data
        self.prompt_data = prompt_data
        self.fn_names = self.get_fn_names()
        self.prompts = self.get_prompts()

    def get_prompts(self) -> list[str]:
        return [prompt["prompt"] for prompt in self.prompt_data]

    def get_fn_names(self) -> list[str]:
        return [fn["name"] for fn in self.function_data]


class CreateTrie:
    def __init__(self, function_data: dict[str, Any],
                 prompt_data: dict[str, str],
                 model: Small_LLM_Model,
                 output_schema: dict[str, Any]) -> None:
        self.function_data = function_data
        self.prompts_raw = prompt_data
        self.output_schema = output_schema
        self.cln = CleanInput(function_data, prompt_data)
        self.function_names = self.cln.fn_names
        self.prompts = self.cln.prompts
        self.Trie = TokenTrie()
        self.model = model
        self.insert_strs()

    def get_wrapped_prompt(self, prompt: str) -> str:
        return (f"Function definitions: {self.function_data}." +
                f"Response output format: {self.output_schema}" +
                "Use the function definitions to answer" +
                "the following prompt in the response output" +
                f" format and nothing more:\n{prompt}\n")

    def make_output_encode(self, prompt, func) -> str:
        wrapped = self.get_wrapped_prompt(prompt)
        wrap_encode = self.model.encode(wrapped)[0].tolist()
        resp_encode = self.model.encode(
            '\n{\n"prompt": "' + prompt + '",\n"name": "' +
            func + '",\n"parameters": '
        )[0].tolist()
        wrap_encode.extend(resp_encode)
        return wrap_encode

    def insert_strs(self):
        for prompt in self.prompts:
            for func in self.function_names:
                self.Trie.add_token_branch(
                            self.make_output_encode(prompt, func)
                            )


class RunModel:
    def __init__(self, prompt: str, model: Small_LLM_Model,
                 trie: TokenTrie, trie_engine: CreateTrie):
        self.model = model
        self.trie = trie
        self.prompt = prompt
        self.trie_engine = trie_engine

    def run_constrained(self) -> str:
        start = datetime.now()
        wrapped_prompt = self.trie_engine.get_wrapped_prompt(self.prompt)
        encoded_prompt = self.model.encode(wrapped_prompt)[0].tolist()
        response_id = len(encoded_prompt)
        while (True):
            print(self.model.decode(encoded_prompt))
            logits = self.model.get_logits_from_input_ids(encoded_prompt)
            allowed_next_tokens = self.trie.get_next_allowed_tokens(
                encoded_prompt)
            if allowed_next_tokens is None:
                break
            max_logit_id = np.argmax(
                self.mask_logits(logits, allowed_next_tokens)).item()
            encoded_prompt.append(max_logit_id)
        end = datetime.now()
        print(f"Time taken to find func name: {(end - start)}")
        return self.model.decode(encoded_prompt)

    @staticmethod
    def mask_logits(logits: list[float], allowed_tokens: list[int]
                    ) -> list[float]:
        masked_set = ~np.isin(np.arange(0, len(logits)), allowed_tokens)
        logits_np = np.array(logits)
        logits_np[masked_set] = float("-inf")
        return list(logits_np)


# class ModelInterface:
#     def __init__(
#             self,
#             ) -> None:
#         self.model = Small_LLM_Model()
#         self.constraints = constraints
#         self.funcs = constraints.function_data
#         self.function_names = self.get_function_names()

#     def get_wrapped_prompt(self, raw_prompt: str) -> tuple[int, str]:
#         wrapped_prompt = (f"Function definitions: {self.funcs}." +
#                           f"Response output format: {
#                               self.constraints.output_schema}" +
#                           "Use the function definitions to answer" +
#                           "the following prompt in the response output" +
#                           f" format and nothing more:\n{raw_prompt}\n")
#         response_start_id = len(wrapped_prompt)
#         wrapped_prompt += '{\n"prompt":"' + f"{raw_prompt}" + '",\n"name": "'
#         return (response_start_id, wrapped_prompt)

#     def run_constrained_model(self, raw_prompt: str):
#         response_start, wrapped_prompt = self.get_wrapped_prompt(raw_prompt)
#         encode = self.model.encode(wrapped_prompt)[0].tolist()
#         token_count = 0
#         start = datetime.now()
#         fn_name_id = len(wrapped_prompt)
#         function_token_ids = [
#             tok for name in self.get_function_name_token_ids() for tok in name]
        
#         end = datetime.now()
#         print(f"Time elapsed: {end - start}")
#         return self.model.decode(encode)[response_start:]

#     def get_clean_vocab_tokens(self) -> dict[int, str]:
#         vocab_file = self.model.get_path_to_vocab_file()
#         with open(vocab_file) as vocab:
#             vocab_dict = json.loads(vocab.read())
#         return {
#             k: v for k, v in vocab_dict.items() if k.isprintable()
#         }

#     def init_fn_name_tree(self, fn_tokens: list[list[int]]) -> None:
#         ## Fix tree first

#     def get_function_name_token_ids(self) -> list[list[int]]:
#         function_token_ids = [
#             self.model.encode(func["name"])[0].tolist() for func in self.funcs]
#         return function_token_ids

#     def get_function_names(self) -> list[str]:
#         return [func["name"] for func in self.funcs]

#     def get_allowed_tokens(self, last_token: int, fn_name_tree: dict[Any]) -> list[int]:

    
