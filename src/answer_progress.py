"""Persist generation progress separately from complete output."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel

from .answerer import Answerer
from .data_models import StudentSearchResults, StudentSearchResultsAndAnswer


def write_json_atomic(path: Path, content: str) -> None:
    """Replace a JSON file only after its complete new content is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class AnswerProgress(BaseModel):
    """Checkpoint answers together with the inputs that produced them."""

    fingerprint: str
    answers: StudentSearchResultsAndAnswer

    @staticmethod
    def input_fingerprint(results: StudentSearchResults, k: int) -> str:
        """Identify questions, source contents and generation settings."""
        settings = {
            "format_version": 1,
            "k": k,
            "model": Answerer.MODEL_NAME,
            "max_new_tokens": Answerer.MAX_NEW_TOKENS,
            "system_prompt": Answerer.SYSTEM_PROMPT,
            "results": results.model_dump(),
        }
        digest = hashlib.sha256(json.dumps(
            settings, sort_keys=True, ensure_ascii=False,
        ).encode("utf-8"))
        paths = sorted({
            source.file_path
            for result in results.search_results
            for source in result.retrieved_sources[:k]
        })
        for name in paths:
            digest.update(name.encode("utf-8"))
            digest.update(hashlib.sha256(Path(name).read_bytes()).digest())
        return digest.hexdigest()

    @classmethod
    def start(
        cls, path: Path, results: StudentSearchResults, k: int, resume: bool,
    ) -> "AnswerProgress":
        """Start fresh or load a checkpoint only when inputs still match."""
        fingerprint = cls.input_fingerprint(results, k)
        if resume:
            progress = cls.model_validate_json(path.read_text("utf-8"))
            if progress.fingerprint != fingerprint:
                raise ValueError(
                    "Cannot resume: questions, sources or settings changed. "
                    "Use another save_directory or restart without --resume."
                )
            completed = progress.answers.search_results
            if progress.answers.k != k or len(completed) > len(
                results.search_results
            ):
                raise ValueError("Invalid answer checkpoint.")
            for answer, question in zip(completed, results.search_results):
                sources = answer.retrieved_sources
                if (
                    answer.question_id != question.question_id
                    or answer.question != question.question
                    or len(sources) > k
                    or sources != question.retrieved_sources[:len(sources)]
                ):
                    raise ValueError("Checkpoint does not match questions.")
            return progress
        return cls(
            fingerprint=fingerprint,
            answers=StudentSearchResultsAndAnswer(search_results=[], k=k),
        )

    def save(self, path: Path) -> None:
        """Persist all completed answers using an atomic replacement."""
        write_json_atomic(path, self.model_dump_json(indent=2))
