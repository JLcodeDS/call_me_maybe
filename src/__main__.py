from .load_input import LoadData
from .data_models import JSONOutput
from .generate_output import ModelInterface, ConstrainModel
import numpy as np


def main() -> None:
    data = LoadData()
    data.load_data()
    constraints = ConstrainModel(
        data.function_data, JSONOutput.model_json_schema())
    model = ModelInterface(constraints)
    outputs: list[str] = []
    for prompt in data.prompt_data:
        outputs.append(model.get_answer_to_prompt(prompt["prompt"]))
    #out = model.get_answer_to_prompt(data.prompt_data[0])
    logits = list(np.arange(0, 10000, 0.5))
    fun_id = [x for sub in model.get_function_name_token_ids() for x in sub]
    logits = model.mask_logits(logits, fun_id)

if __name__ == "__main__":
    # try:
    #    main()
    # except Exception as e:
    #    print(e)
   main()
