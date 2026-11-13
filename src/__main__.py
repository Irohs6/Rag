import fire
from pathlib import Path


def index(data_dir: str = "data/raw") -> None:
    for path in Path(data_dir).rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".txt"}:
            print(path)


if __name__ == "__main__":
    fire.Fire({"index": index})
