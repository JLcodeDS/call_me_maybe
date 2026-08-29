from .load_input import LoadData
from .data_models import JSONOutput
from .generate_output import CreateFnTrie, RunModel
from llm_sdk import Small_LLM_Model  # type:ignore


def main() -> None:
    data = LoadData()
    data.load_data()
    model = Small_LLM_Model()
    output_schema = JSONOutput.model_json_schema()
    trie_engine = CreateFnTrie(data.function_data, model)
    prompts = [p["prompt"] for p in data.prompt_data]
    run = RunModel(prompts[0], model, trie_engine.Trie,
                   output_schema, data.function_data)
    output = run.run_constrained()
    print(output)

if __name__ == "__main__":
    # try:
    #    main()
    # except Exception as e:
    #    print(e)
   main()
