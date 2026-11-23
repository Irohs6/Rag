from pathlib import Path
from tqdm import tqdm
from .chunker import Chunker
from .data_models import (
    MinimalSource,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
)
from uuid import uuid4
from .retriever import Retriever


class RagCli:

    @staticmethod
    def index(
        data_dir: str = "data/raw",
        max_chunk_size: int = 2000,
        save_directory: str = "data/processed/",
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
                    str(path), text, max_chunk_size, overlap=50
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

        retriever = Retriever()
        retriever.index(all_chunks)
        retriever.save(save_directory)

        print(f"Index et sources enregistrés dans {save_directory}")

    @staticmethod
    def search(
        query: str,
        index_directory: str = "data/processed",
        k: int = 5,
    ) -> MinimalSearchResults:

        retriever = Retriever()
        retriever.load(index_directory)

        sources = retriever.search(query, k=k)

        return MinimalSearchResults(
            question_id=str(uuid4()),
            question=query,
            retrieved_sources=sources,
        )

    @staticmethod
    def search_dataset(
        dataset_path: str,
        k: int = 10,
        index_directory: str = "data/processed",
        save_directory: str = "data/output/search_results",
    ) -> None:
        # Charger le dataset contenant les questions RAG.
        dataset_file = Path(dataset_path)
        dataset = RagDataset.model_validate_json(
            dataset_file.read_text(encoding="utf-8")
        )

        retriever = Retriever()
        retriever.load(index_directory)
        results: list[MinimalSearchResults] = []

        for question in tqdm(
            dataset.rag_questions,
            desc="Searching questions",
            unit="questions",
        ):
            sources = retriever.search(
                query=question.question,
                k=k,
            )

            results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
            )

        # Regrouper tous les résultats dans le format attendu.
        output = StudentSearchResults(
            search_results=results,
            k=k,
        )

        # Enregistrer sous le même nom que le dataset d'entrée.
        output_directory = Path(save_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        output_file = output_directory / dataset_file.name
        output_file.write_text(
            output.model_dump_json(indent=2),
            encoding="utf-8",
        )

        print(f"Résultats enregistrés dans {output_file}")
