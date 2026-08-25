from load_input import LoadData
from data_models import CreateResponseModel
#from .generate_output import GenerateOutput


def main() -> None:
    data = LoadData()
    data.load_data()
    crm = CreateResponseModel(data.function_data)
    ModelResponse = crm.create_schema()
    schema =  {
"prompt": "Reverse the string 'hello'",
"name": "fn_reverse_string",
"parameters": {"s": "hello"}
}
    ModelResponse.model_validate(
       schema
    )

    #gen = GenerateOutput("What is the sum of 2 and 3?")
    #gen.run_model(data.function_data)


if __name__ == "__main__":
    #try:
    #    main()
    #except Exception as e:
    #    print(e)
    main()