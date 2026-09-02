from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
from typing import Any
#  from datetime import datetime
from .data_models import TokenTrie
from .data_preparation import CreateTries, PrepareFnData
import re
import json
from .fsms import FSMDispatch
### Change structure ###
## Query first function name through Trie and save
## Then Query params and save
## Finally Generate the actual JSON output with string and that

# Pre sort tokens will be the longest part 


class CompileParamMask:
    def __init__(self, function_data: list[dict[str, Any]],
                 model: Small_LLM_Model):
        self.function_data = function_data
        self.json_to_regex = {
            "string": re.compile(r'(^")|((^\})",)'),
            "number": re.compile(r'-?\d+(?:,\s)?'),
            "boolean": re.compile(r'^(true|false)(,)?$'),
            "null": re.compile(r'^null([,}])?$'),
            "integer": re.compile(r'^-?(0|[1-9]\d*)(,)?$'),
            "array": re.compile(r'^\[\s*[^\[\]]*\s*\](,)?$'),
            "object": re.compile(r'^\{\s*[^\{\}]*\s*\}(,)?$')
        }
#  Check if all this shit is even the good method (look into fsm & pushdown etc advantages and all)
        self.type_eof = {
            "string": re.compile(r'",'),
            "number": re.compile(r'\d,$'),
            "boolean": re.compile(r'(?:true|false)\s*,?\s*$'),
            "null": re.compile(r'null\s*,?\s*$'),
            "integer": re.compile(r'\d\s*,?\s*$'),
            "array": re.compile(r']\s*,?\s*$'),
            "object": re.compile(r'}\s*,?\s*$')
        }

        self.save_fn_param_types()
        self.model = model
        self.save_type_tokens()

    def save_type_tokens(self) -> None:
        self.type_to_token_mask = {
            k: self.get_type_tokens(k) for k in self.json_to_regex
        }

    def save_fn_param_types(self) -> None:
        self.fn_param_types = {
            fn["name"]: {
                k: v["type"] for k, v in fn["parameters"].items()
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
        token_matches = [v for k, v in vocab_json.items(
            ) if r.match(k.replace("Ġ", " "))]
        return token_matches


class RunModel:
    def __init__(self, prompt: str, model: Small_LLM_Model,
                 trie: TokenTrie, output_schema: dict[str, Any],
                 clean_fn_data: PrepareFnData, regex_map: CompileParamMask):
        self.model = model
        self.trie = trie
        self.prompt = prompt
        self.output_schema = output_schema
        self.clean_fn_data = clean_fn_data
        self.regex_map = regex_map
        self.parameters: dict[str, Any] = {}

    def find_fn_name(self) -> None:
        fn_prompt = self.get_wrapped_prompt()
        self.response_id = len(fn_prompt)
        fn_prompt += '{"prompt": "' + self.prompt + ', "name": "'
        encoded_prompt = self.model.encode(fn_prompt)[0].tolist()
        encoded_fn: list[int] = []
        while True:
            logits = self.model.get_logits_from_input_ids(encoded_prompt)
            allowed_next_tokens = self.trie.get_next_allowed_tokens(
                encoded_fn)
            if allowed_next_tokens is None:
                break
            max_logit_id = np.argmax(
                self.mask_logits(logits, allowed_next_tokens)).item()
            encoded_prompt.append(max_logit_id)
            encoded_fn.append(max_logit_id)
        self.function_name = self.model.decode(encoded_fn)

    def find_params(self) -> Any:
        params = self.clean_fn_data.fn_params[self.function_name]
        param_prompt = self.get_param_prompt()
        encoded_param: list[int] = []
        for p, p_type in params.items():
            param_prompt += '"' + p + '": '
            encoded_response = self.model.encode(param_prompt)[0].tolist()
            fsm = FSMDispatch(p_type)
        #fn_name = prompt_w_fn_name[fn_id:]
        #params = self.regex_map.fn_param_types[fn_name]
        #prompt_w_fn_name += '", "parameters": {'
        
        #    encoded_response = self.model.encode(prompt_w_fn_name)[0].tolist()
        #    encoded_param = []
        #    decoded_param = ""
        #    while not self.regex_map.type_eof[p_type].match(decoded_param):
        #        allowed_tokens = self.regex_map.get_fn_param_tokens(fn_name, p)
        #        logits = self.model.get_logits_from_input_ids(encoded_response)
        #        max_logit_id = np.argmax(
        #            self.mask_logits(logits, allowed_tokens)).item()
        #        encoded_param.append(max_logit_id)
        #        encoded_response.append(max_logit_id)
        #        decoded_param = self.model.decode(encoded_param)
        #        print(self.model.decode(encoded_response)[response_id:])
        #    prompt_w_fn_name += self.model.decode(encoded_param)
        #prompt_w_fn_name += '}}'
        #return prompt_w_fn_name[response_id:]

    def get_wrapped_prompt(self) -> str:
        return ("You are a function-calling assistant.\n"
                f"Available functions:\n{self.clean_fn_data.fn_names_defs}\n" +
                f"User request: {self.prompt}\n" +
                "Output the call in JSON format matching the schema:\n"
                '{"prompt": <prompt>, "name": "<function_name>", "parameters": {<param_name>: <param_value>, ...}}\n'
                "JSON output:\n")

    def get_param_prompt(self) -> str:
        return (
            self.get_wrapped_prompt() +
            '{"prompt": "' + self.prompt +
            ', "name": "' + self.function_name + '", "parameters": {'
        )
    
    @staticmethod
    def mask_logits(logits: list[float], allowed_tokens: list[int]
                    ) -> list[float]:
        masked = [float("-inf")] * len(logits)
        for token_id in allowed_tokens:
            masked[token_id] = logits[token_id]
        return masked
