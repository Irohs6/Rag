import fire
from pathlib import Path
from data_models import MinimalSource
from chuncker import chunk_markdown, chunk_python


def index(data_dir: str = "data/raw", max_chunck_size: int = 2000) -> None:
    all_chunks = []
    for path in Path(data_dir).rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8")

            if path.suffix == ".py":
                sources: list[MinimalSource] = chunk_python(
                    str(path), text, max_chunck_size)
            else:
                sources: list[MinimalSource] = chunk_markdown(
                    str(path), text, max_chunck_size)
            for source in sources:
                passage = text[
                    source.first_character_index:source.last_character_index]
                all_chunks.extend((passage, source))


if __name__ == "__main__":
    fire.Fire(index)
