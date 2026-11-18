from pathlib import Path
import bm25s
from tqdm import tqdm
import re
import json
from .chunker import Chunker
from .data_models import (
    MinimalSource,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
)


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

        token_chunks = []
        for passage, min_source in all_chunks:
            if Path(min_source.file_path).suffix == ".py":
                tokens = RagCli.tokenizer_code_avancé(passage)
            else:
                tokens = bm25s.tokenize(
                    passage,
                    return_ids=False,
                    show_progress=False,
                )[0]
            token_chunks.append(tokens)
        retriever = bm25s.BM25()
        retriever.index(token_chunks)

        output_directory = Path(save_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        # Sauvegarder l’index utilisé par BM25.
        retriever.save(str(output_directory / "bm25"))

        # Sauvegarder les coordonnées des chunks dans le même ordre.
        sources = [
            source.model_dump()
            for _, source in all_chunks
        ]

        sources_file = output_directory / "sources.json"
        sources_file.write_text(
            json.dumps(sources, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"Index et sources enregistrés dans {output_directory}")
        RagCli.search_dataset(
                dataset_path=(
                    "data/datasets/UnansweredQuestions/"
                    "dataset_docs_public.json"
                ),
                retriever=retriever,
                all_chunks=all_chunks,
            )

    @staticmethod
    def search(
        query: str,
        index_directory: str = "data/processed",
        k: int = 5,
    ) -> MinimalSearchResults:
        # Liste de mots avec le tokenizer de base.
        tokens_base = bm25s.tokenize(
            query,
            return_ids=False,
            show_progress=False,
        )[0]

        # Liste de mots avec ton tokenizer pour le code.
        tokens_code = RagCli.tokenizer_code_avancé(query)

        # Regrouper les mots sans doublons.
        tokens = list(dict.fromkeys(tokens_base + tokens_code))

        # Une liste de questions, contenant ici une seule question.
        query_tokens = [tokens]

        indices, scores = retriever.retrieve(
            query_tokens,
            k=min(k, len(all_chunks)),
        )

        sources = []
        for chunk_index in indices[0]:
            passage, source = all_chunks[int(chunk_index)]
            sources.append(source)

        return sources

    @staticmethod
    def search_dataset(
        dataset_path: str,
        k: int = 10,
        index_directory: str = "data/processed",
        save_directory: str = "data/output/search_results",
    ) -> None:
        # Lire les questions et les convertir en objets Pydantic.
        dataset_file = Path(dataset_path)
        dataset = RagDataset.model_validate_json(
            dataset_file.read_text(encoding="utf-8")
        )

        results = []

        for question in tqdm(
            dataset.rag_questions,
            desc="Recherche des sources",
            unit="question",
        ):
            sources = RagCli.search(
                query=question.question,
                retriever=retriever,
                all_chunks=all_chunks,
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
