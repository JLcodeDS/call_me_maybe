from .load_input import LoadData
from .data_models import DynamicParameterModel
from .generate_output import RunModel
from .data_preparation import PrepareFnData, BuildTokenCache, CreateTries
from llm_sdk import Small_LLM_Model  # type:ignore
from datetime import datetime
import json


def main() -> None:
    data = LoadData()
    data.load_data()
    model = Small_LLM_Model(device="cuda")
    clean_data = PrepareFnData(data.function_data)
    dynamic_model = DynamicParameterModel(clean_data.fn_params)
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
    with open(data.dict_params["output"], 'w+') as f:
        json.dump(outputs, f)
        print(f"Function calls saved at {data.dict_params["output"]}.")
    end = datetime.now()
    print(f"Time elapsed: {end - start}")


if __name__ == "__main__":
    # try:
    #    main()
    # except Exception as e:
    #    print(e)
    main()
