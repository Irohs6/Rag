from pathlib import Path

import bm25s
import fire
from tqdm import tqdm

from .chunker import Chunker
from .data_models import MinimalSource


def index(
    data_dir: str = "data/raw",
    max_chunk_size: int = 2000,
) -> None:
    all_chunks: list[tuple[str, MinimalSource]] = []

    files = [
        path
        for path in Path(data_dir).rglob("*")
        if path.is_file() and path.suffix in {".py", ".md", ".txt"}
    ]

    for path in tqdm(files, desc="Découpage des fichiers", unit="fichier"):
        text = path.read_text(encoding="utf-8")

        if path.suffix == ".py":
            sources = Chunker.chunk_python(
                str(path), text, max_chunk_size
            )
        else:
            sources = Chunker.chunk_markdown(
                str(path), text, max_chunk_size
            )

        for source in sources:
            passage = text[
                source.first_character_index:source.last_character_index
            ]
            all_chunks.append((passage, source))

    if not all_chunks:
        print("Aucun chunk créé.")
        return

    print(f"{len(all_chunks)} chunks créés")


if __name__ == "__main__":
    fire.Fire({"index": index})
