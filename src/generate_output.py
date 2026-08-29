from llm_sdk import Small_LLM_Model
import numpy as np
from typing import Any, Pattern
from datetime import datetime
from .data_models import TokenTrie
import re
import json


#class CleanInput:
#    def __init__(self, function_data: dict[str, Any],
#                 prompt_data: dict[str, str]) -> None:
#        self.function_data = function_data
#        self.prompt_data = prompt_data
#        self.fn_names = self.get_fn_names()
#        self.prompts = self.get_prompts()

#    def get_prompts(self) -> list[str]:
#        return [prompt["prompt"] for prompt in self.prompt_data]

#    def get_fn_names(self) -> list[str]:
#        return [fn["name"] for fn in self.function_data]


class CreateFnTrie:
    def __init__(self, function_data: dict[str, Any],
                 model: Small_LLM_Model) -> None:
        self.function_data = function_data
        self.function_names = self.get_fn_names()
        self.Trie = TokenTrie()
        self.model = model
        self.insert_strs()

    def make_encode(self, func) -> str:
        return self.model.encode(func)[0].tolist()

    def insert_strs(self):
        for func in self.function_names:
            self.Trie.add_token_branch(
                        self.make_encode(func)
                        )

    def get_fn_names(self) -> list[str]:
        return [fn["name"] for fn in self.function_data]


class CompileParamMask:
    def __init__(self, function_data: list[dict[str, Any]],
                 model: Small_LLM_Model):
        self.function_data = function_data
        self.json_to_regex = {
            "string": re.compile(r'^"[^"\\]*(\\.[^"\\]*)*"([,}])?$'),
            "number": re.compile(r'^-?(0|[1-9]\d*)(\.\d+)?([,}])?$'),
            "boolean": re.compile(r'^(true|false)([,}])?$'),
            "null": re.compile(r'^null([,}])?$'),
            "integer": re.compile(r'^-?(0|[1-9]\d*)([,}])?$'),
            "array": re.compile(r'^\[\s*[^\[\]]*\s*\]([,}])?$'),
            "object": re.compile(r'^\{\s*[^\{\}]*\s*\}([,}])?$')
        }

        self.save_fn_param_types()
        self.model = model
    
    def save_type_tokens(self) -> None:
        self.type_to_token_mask = {
            k: self.get_type_tokens(k) for k in self.json_to_regex
        }

    def save_fn_param_types(self) -> None:
        self.fn_param_types = {
            fn["name"]: {
                k: k["type"] for k in fn["parameters"]
            } for fn in self.function_data
        }

    def get_fn_param_tokens(self, fn: str, param: str) -> list[int]:
        return self.type_to_token_mask[
            self.fn_param_types[fn][param]
        ]

    def get_type_tokens(self, param_type: str) -> list[int]:
        with open(self.model.get_path_to_vocab_file()) as vocab:
            vocab_json = json.load(vocab)
        r = self.json_to_regex[param_type]
        token_matches = [k for k, v in vocab_json.items() if r.match(v)]
        return token_matches


class RunModel:
    def __init__(self, prompt: str, model: Small_LLM_Model,
                 trie: TokenTrie, output_schema: dict[str, Any],
                 function_data: dict[str, Any], regex_map: CompileParamMask):
        self.model = model
        self.trie = trie
        self.prompt = prompt
        self.output_schema = output_schema
        self.function_data = function_data
        self.regex_map = regex_map

    def run_constrained(self) -> str:
        wrapped_prompt = self.get_wrapped_prompt(self.prompt)
        response_id = len(wrapped_prompt)
        wrapped_prompt += '{\n"prompt": "' + self.prompt + '",\n"name": "'
        encoded_prompt = self.model.encode(wrapped_prompt)[0].tolist()
        encoded_fn: list[int] = []
        while (True):
            logits = self.model.get_logits_from_input_ids(encoded_prompt)
            allowed_next_tokens = self.trie.get_next_allowed_tokens(
                encoded_fn)
            if allowed_next_tokens is None:
                break
            max_logit_id = np.argmax(
                self.mask_logits(logits, allowed_next_tokens)).item()
            encoded_prompt.append(max_logit_id)
            encoded_fn.append(max_logit_id)
        prompt_w_fn_name = self.model.decode(encoded_prompt)
        fn_name = prompt_w_fn_name[response_id:]
        params = self.regex_map.fn_param_types[fn_name]
        prompt_w_fn_name += '",\n"parameters": {'
        for p, p_type in params.items():
            prompt_w_fn_name += '"' + p + '": '
            param_id = len(prompt_w_fn_name)
            encoded_response = self.model.encode(prompt_w_fn_name)[0].tolist()
            encoded_param = []
            while(WHAT):
                allowed_tokens = self.regex_map.get_fn_param_tokens(fn_name, p)
                logits = self.model.get_logits_from_input_ids(encoded_response)
                max_logit_id = np.argmax(
                self.mask_logits(logits, allowed_tokens)).item()
                encoded_param.append(max_logit_id)
                encoded_response.append(max_logit_id)
            prompt_w_fn_name += self.model.decode(encoded_param)
            # think of condition tomorrow?
            ### how do I stop it??? ffs
        return prompt_w_fn_name[response_id:]

    def get_wrapped_prompt(self, prompt: str) -> str:
        return (f"Function definitions: {self.function_data}." +
                f"Response output format: {self.output_schema}" +
                "Use the function definitions to answer" +
                "the following prompt in the response output" +
                f" format and nothing more:\n{prompt}\n")

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

    
