from .load_input import LoadData
from .data_models import JSONOutput
from .generate_output import RunModel
from .data_preparation import PrepareFnData, BuildTokenCache, CreateTries
from llm_sdk import Small_LLM_Model  # type:ignore
from datetime import datetime
import json


def main() -> None:
    data = LoadData()
    data.load_data()
    model = Small_LLM_Model()
    clean_data = PrepareFnData(data.function_data)
    output_schema = JSONOutput.model_json_schema()
    trie_engine = CreateTries(data.function_data, model)
    with open(model.get_path_to_vocab_file(), 'r', encoding='utf-8') as v:
        vocab = json.load(v)
    token_start = datetime.now()
    token_cache = BuildTokenCache(vocab)
    token_cache.build_cache()
    cache = token_cache.token_cache
    print(
        [model.decode(tok) for tok in cache["START_STR"]]
    )
    print(f"Cache built in {datetime.now() - token_start}")
    prompts = [p["prompt"] for p in data.prompt_data]
    start = datetime.now()
    for i, prompt in enumerate(prompts):
        print(f"Evaluating prompt: {prompt}")
        run = RunModel(prompt, model, trie_engine.FnTrie,
                       output_schema, clean_data, cache)
        s = run.run()
        print(f"Done:\n{s}\n")
    end = datetime.now()
    print(f"Time elapsed: {end - start}")


if __name__ == "__main__":
    # try:
    #    main()
    # except Exception as e:
    #    print(e)
    main()
