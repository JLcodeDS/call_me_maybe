from .load_input import LoadData
from .generate_output import GenerateOutput


def main() -> None:
    data = LoadData()
    data.load_data()
    print(data.prompt_data)
    print(data.function_data)
    gen = GenerateOutput("What is the sum of 2 and 3?")
    gen.run_model()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
