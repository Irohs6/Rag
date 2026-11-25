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
from .answerer import Answerer
from .answer_progress import AnswerProgress, write_json_atomic


class RagCli:

    @staticmethod
    def index(
        data_dir: str = "data/raw",
        max_chunk_size: int = 2000,
        save_directory: str = "data/processed/",
    ) -> None:

        if not Path(data_dir).is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than 0")

        if max_chunk_size > 2000:
            raise ValueError("max_chunk_size must not exceed 2000")

        all_chunks: list[tuple[str, MinimalSource]] = []

        files = [
            path
            for path in Path(data_dir).rglob("*")
            if path.is_file() and path.suffix in {".py", ".md", ".txt"}
        ]

        for path in tqdm(files, desc="Chunking files", unit="files"):
            text = path.read_text(encoding="utf-8")

            if path.suffix == ".py":
                sources = Chunker.chunk_python(str(path), text, max_chunk_size)
            else:
                sources = Chunker.chunk_markdown(
                    str(path), text, max_chunk_size, overlap=50
                )

            for source in sources:
                passage = text[
                    source.first_character_index: source.last_character_index
                ]
                all_chunks.append((passage, source))

        if not all_chunks:
            print("No chunks created.")
            return
        print(f"{len(all_chunks)} chunks created.")

        retriever = Retriever()
        retriever.index(all_chunks)
        retriever.save(save_directory)

        print(f"Index and sources saved in {save_directory}")

    @staticmethod
    def search(
        query: str,
        index_directory: str = "data/processed",
        k: int = 5,
    ) -> MinimalSearchResults:

        if query.strip() == "":
            raise ValueError("Query must not be empty")
        if k < 1:
            raise ValueError("k must be greater than 0")
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

        if not Path(dataset_path).is_file():
            raise ValueError(f"Dataset file not found: {dataset_path}")

        if k < 1:
            raise ValueError("k must be greater than 0")

        if not Path(index_directory).is_dir():
            raise ValueError(f"Index directory not found: {index_directory}")
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

        print(f"Search results saved to {output_file}")

    @staticmethod
    def answer(
        query: str,
        index_directory: str = "data/processed",
        k: int = 5,
    ) -> str:
        """Retrieve sources and generate one answer within Qwen's budget."""
        results = RagCli.search(query, index_directory=index_directory, k=k)
        return Answerer().answer(results, k=k).answer

    @staticmethod
    def answer_dataset(
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        k: int = 5,
        resume: bool = False,
    ) -> None:
        """Generate answers, checkpoint each result and optionally resume."""
        if k < 1:
            raise ValueError("k must be greater than 0")
        input_file = Path(student_search_results_path)
        results = StudentSearchResults.model_validate_json(
            input_file.read_text(encoding="utf-8")
        )
        output_file = Path(save_directory) / input_file.name
        if output_file.resolve() == input_file.resolve():
            raise ValueError("Output must not overwrite search results.")
        checkpoint_file = output_file.with_suffix(".progress.json")
        if checkpoint_file.resolve() == input_file.resolve():
            raise ValueError("Checkpoint must not overwrite search results.")
        progress = AnswerProgress.start(checkpoint_file, results, k, resume)
        progress.save(checkpoint_file)
        completed = len(progress.answers.search_results)
        answerer = Answerer()

        for search_result in tqdm(
            results.search_results[completed:],
            initial=completed,
            total=len(results.search_results),
            desc="Generating answers",
            unit="answer",
            dynamic_ncols=True,
        ):
            answer = answerer.answer(search_result, k=k)
            progress.answers.search_results.append(answer)
            progress.save(checkpoint_file)

        write_json_atomic(output_file, progress.answers.model_dump_json(
            indent=2,
        ))
        print(f"Answers saved to {output_file}")
