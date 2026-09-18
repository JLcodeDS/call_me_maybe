"""Main orchestration module for the LLM inference pipeline.

This script coordinates loading input data, preparing
parameters, generating the token cache, and executing
prompts against a small language model (Small_LLM_Model)
to produce and save structured function calls.
"""

from .load_input import LoadData
from .data_models import DynamicParameterModel
from .generate_output import RunModel
from .data_preparation import PrepareFnData, BuildTokenCache, CreateTries
from llm_sdk import Small_LLM_Model  # type:ignore
from datetime import datetime
import json
import os


def main() -> None:
    """Executes the complete data processing and prompt evaluation workflow.

    The function performs the following steps:
    1. Initializes components and loads input configuration data.
    2. Cleans function definitions and builds dynamic parameter models.
    3. Constructs prefix trees (Tries) and indexes the vocabulary tokens.
    4. Evaluates each input prompt sequentially through the LLM.
    5. Exports the compiled function-call outputs to the designated JSON file.

    Raises:
        Exception: If any error occurs during the execution
        of the data loading, model inference, or file writing processes.
    """
    data = LoadData()
    data.load_data()
    model = Small_LLM_Model()
    clean_data = PrepareFnData(data.function_data)
    dynamic_model = DynamicParameterModel(clean_data.json_params)
    trie_engine = CreateTries(data.function_data, model)
    with open(model.get_path_to_vocab_file(), 'r', encoding='utf-8') as v:
        vocab = json.load(v)
    token_start = datetime.now()
    token_cache = BuildTokenCache(vocab)
    token_cache.build_cache()
    cache = token_cache.token_cache
    print(f"Cache built in {datetime.now() - token_start}")
    prompts = [p["prompt"] for p in data.prompt_data]
    start = datetime.now()
    outputs = []
    for i, prompt in enumerate(prompts):
        print(f"Evaluating prompt: {prompt}")
        run = RunModel(prompt, model, trie_engine.FnTrie,
                       clean_data, dynamic_model, cache)
        s = run.run()
        outputs.append(s)
        print(f"Done:\n{s}\n")
    output_dir = os.path.dirname(data.dict_params["output"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(data.dict_params["output"], 'w+') as f:
        json.dump(outputs, f, indent=4)
        print(f"Function calls saved at {data.dict_params['output']}.")
    end = datetime.now()
    print(f"Time elapsed: {end - start}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
