import json
from pathlib import Path

import bm25s
from tqdm import tqdm

from .data_models import MinimalSource
from .tokenizer import Tokenizer


class Retriever:
    """Build, persist and search a BM25 index with its source locations."""

    def __init__(self) -> None:
        """Initialize an empty retriever."""
        self.bm25_index: bm25s.BM25 | None = None
        self.sources: list[MinimalSource] = []

    def index(
        self,
        chunks: list[tuple[str, MinimalSource]],
    ) -> None:
        """Tokenize chunks and build their BM25 index."""
        if not chunks:
            raise ValueError("Not enough chunks to index.")

        token_chunks: list[list[str]] = []

        for passage, source in tqdm(
            chunks,
            desc="Tokenizing chunks",
            unit="chunk",
        ):
            tokens = Tokenizer.tokenize_document(
                passage,
                source.file_path,
            )
            token_chunks.append(tokens)

        if not any(token_chunks):
            raise ValueError("No tokens to index after tokenization.")

        bm25_index = bm25s.BM25()
        bm25_index.index(token_chunks)

        self.bm25_index = bm25_index
        self.sources = [source for _, source in chunks]

    def save(self, directory: str) -> None:
        """Save the BM25 index and its ordered source locations."""
        if self.bm25_index is None:
            raise ValueError("No index to save.")

        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        self.bm25_index.save(str(output_directory / "bm25"))

        sources_data = [
            source.model_dump()
            for source in self.sources
        ]

        sources_file = output_directory / "sources.json"
        sources_file.write_text(
            json.dumps(sources_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, directory: str) -> None:
        """Load a saved BM25 index and validate its source locations."""
        index_directory = Path(directory)
        sources_file = index_directory / "sources.json"

        sources_data = json.loads(
            sources_file.read_text(encoding="utf-8")
        )

        if not isinstance(sources_data, list) or not sources_data:
            raise ValueError("Invalid or empty sources list.")

        sources = [
            MinimalSource.model_validate(source)
            for source in sources_data
        ]

        bm25_index = bm25s.BM25.load(
            str(index_directory / "bm25"),
            load_corpus=False,
        )

        self.bm25_index = bm25_index
        self.sources = sources

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[MinimalSource]:
        """Return the source locations of the top-k matching chunks."""
        if self.bm25_index is None:
            raise ValueError("No index available for search.")

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        if not query.strip():
            raise ValueError("Query must not be empty.")

        tokens = Tokenizer.tokenize_query(query)

        if not tokens:
            return []

        indices, _ = self.bm25_index.retrieve(
            [tokens],
            k=min(k, len(self.sources)),
        )

        return [
            self.sources[int(chunk_index)]
            for chunk_index in indices[0]
        ]
