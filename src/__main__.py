from .load_input import LoadData
from .data_models import JSONOutput
from .generate_output import GenerateOutput, ConstrainModel


def main() -> None:
    data = LoadData()
    data.load_data()
    constraints = ConstrainModel(
        data.function_data, JSONOutput.model_json_schema())
    gen = GenerateOutput(data.prompt_data, constraints)
    out = gen.get_answer_to_prompt(data.prompt_data[0])
    print(out)

if __name__ == "__main__":
    # try:
    #    main()
    # except Exception as e:
    #    print(e)
   main()
