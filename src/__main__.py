from pathlib import Path

import bm25s
import fire
from tqdm import tqdm
import re
from .chunker import Chunker
from .data_models import (
    MinimalSource,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
)

KEYWORDS_PYTHON: set[str] = {
    "def", "class", "import", "from", "return",
    "if", "else", "elif", "for", "while",
    "try", "except", "with", "as", "pass",
    "none", "true", "false", "and", "or", "not",
    "in", "is", "lambda", "yield", "async",
    "await", "self",
}


def tokenizer_code_avancé(
    texte: str,
    exclure_keywords: set = KEYWORDS_PYTHON,
    min_len: int = 3
) -> list[str]:
    """
    Tokenizer code avancé avec exclusion des mots-clés du langage.
    Idéal quand on indexe des fonctions Python / JS.
    """
    # Séparer camelCase / PascalCase
    texte = re.sub(r'([a-z])([A-Z])', r'\1 \2', texte)
    texte = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', texte)

    # Remplacer les séparateurs
    texte = re.sub(r'[_\-./\\:@#()\[\]{}<>,;=!?|&*+%^~`"\']+', ' ', texte)

    # Tokeniser
    tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', texte)
    tokens = [t.lower() for t in tokens]

    # Filtrer
    return [
        t for t in tokens
        if len(t) >= min_len and t not in exclure_keywords
    ]


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
    texts = [passage for passage, source in all_chunks]

    token_chunks = [tokenizer_code_avancé(text) for text in texts]
    retriever = bm25s.BM25()
    retriever.index(token_chunks)

    search_dataset(
            dataset_path=(
                "data/datasets/UnansweredQuestions/"
                "dataset_code_public.json"
            ),
            retriever=retriever,
            all_chunks=all_chunks,
        )


def search(
    query: str,
    retriever: bm25s.BM25,
    all_chunks: list[tuple[str, MinimalSource]],
    k: int = 5,
) -> list[MinimalSource]:
    query_tokens = [tokenizer_code_avancé(query)]

    indices, scores = retriever.retrieve(
        query_tokens,
        k=min(k, len(all_chunks)),
    )

    sources = []
    for chunk_index in indices[0]:
        passage, source = all_chunks[int(chunk_index)]
        sources.append(source)

    return sources


def search_dataset(
    dataset_path: str,
    retriever: bm25s.BM25,
    all_chunks: list[tuple[str, MinimalSource]],
    save_directory: str = "data/output/search_results",
    k: int = 10,
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
        sources = search(
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


if __name__ == "__main__":
    fire.Fire({"index": index})
