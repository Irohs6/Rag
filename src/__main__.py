import fire
from .rag_cli import RagCli
from json import JSONDecodeError


if __name__ == "__main__":
    try:
        fire.Fire(RagCli())
    except (ValueError, JSONDecodeError, FileNotFoundError) as e:
        print(f"Error: {e}")
