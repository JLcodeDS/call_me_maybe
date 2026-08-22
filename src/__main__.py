from .load_input import LoadData


def main() -> None:
    data = LoadData()
    data.load_data()
    print(data.prompt_data)
    print(data.function_data)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
