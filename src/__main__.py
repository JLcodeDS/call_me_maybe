from .load_input import LoadData
from .data_models import JSONOutput
from .generate_output import CreateFnTrie, RunModel
from llm_sdk import Small_LLM_Model  # type:ignore
from datetime import datetime


def main() -> None:
    data = LoadData()
    data.load_data()
    model = Small_LLM_Model()
    output_schema = JSONOutput.model_json_schema()
    trie_engine = CreateFnTrie(data.function_data, model)
    prompts = [p["prompt"] for p in data.prompt_data]
    start = datetime.now()
    for prompt in prompts:
        run = RunModel(prompt, model, trie_engine.Trie,
                       output_schema, data.function_data)
        output = run.run_constrained()
        print(output + '",\n"parameters:": ')
    end = datetime.now()
    print(f"Time elapsed: {end - start}")


if __name__ == "__main__":
    # try:
    #    main()
    # except Exception as e:
    #    print(e)
    main()
