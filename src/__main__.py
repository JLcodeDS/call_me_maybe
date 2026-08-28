from .load_input import LoadData
from .data_models import JSONOutput
from .generate_output import CleanInput, CreateTrie, RunModel
from llm_sdk import Small_LLM_Model  # type:ignore


def main() -> None:
    data = LoadData()
    data.load_data()
    model = Small_LLM_Model()
    clean = CleanInput(data.function_data, data.prompt_data)
    output_schema = JSONOutput.model_json_schema()
    trie_engine = CreateTrie(data.function_data, data.prompt_data,
                             model, output_schema)
    prompts = clean.prompts
    run = RunModel(prompts[0], model, trie_engine.Trie, trie_engine)
    output = run.run_constrained()
    print(output)

if __name__ == "__main__":
    # try:
    #    main()
    # except Exception as e:
    #    print(e)
   main()
