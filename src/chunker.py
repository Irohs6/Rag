import ast

from .data_models import MinimalSource
import re


class Chunker:
    """Split source text while preserving original character coordinates."""

    @staticmethod
    def chunk_markdown(
        file_path: str,
        text: str,
        max_chunk_size: int,
        overlap: int,
    ) -> list[MinimalSource]:
        """Split text near punctuation with a word-based overlap."""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size doit être supérieur à 0")

        if overlap < 0:
            raise ValueError("overlap doit être supérieur ou égal à 0")

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_chunk_size, len(text))

            # Si tout le texte restant tient, inutile de chercher un point.
            if end < len(text):
                last_dot = text.rfind(".", start, end)

                if last_dot != -1:
                    end = last_dot + 1

            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                )
            )
            if end == len(text):
                break

            next_start = end

            if overlap > 0:
                word = list(re.finditer(r"\S+", text[start:end]))
                if len(word) > overlap:
                    next_start = start + word[-overlap].start()

            start = next_start

        return chunks

    @staticmethod
    def chunk_python(
        file_path: str,
        text: str,
        max_chunk_size: int,
    ) -> list[MinimalSource]:
        """Keep functions/classes and also index the text between them.

        Imports, constants, module docstrings and outside comments are kept
        in the gaps. Unsupported Python syntax falls back to fixed slices.
        All coordinates use an exclusive end in the original text.
        """
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size doit être supérieur à 0")

        try:
            parse_code = ast.parse(text)
        except SyntaxError:
            return Chunker._split_span(
                file_path, text, 0, len(text), max_chunk_size,
            )

        line_starts = [0]
        for line in text.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(line))

        chunks: list[MinimalSource] = []
        previous_end = 0
        for element in parse_code.body:
            if not isinstance(
                element,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            first_line = element.lineno
            if element.decorator_list:
                first_line = element.decorator_list[0].lineno
            if element.end_lineno is None:
                continue
            start = line_starts[first_line - 1]
            end = line_starts[element.end_lineno]
            chunks.extend(Chunker._split_span(
                file_path, text, previous_end, start, max_chunk_size,
            ))
            chunks.extend(Chunker._split_span(
                file_path, text, start, end, max_chunk_size,
            ))
            previous_end = end

        chunks.extend(Chunker._split_span(
            file_path, text, previous_end, len(text), max_chunk_size,
        ))
        return chunks

    @staticmethod
    def _split_span(
        file_path: str, text: str, start: int, end: int, max_size: int,
    ) -> list[MinimalSource]:
        """Slice a region into bounded chunks, skipping whitespace."""
        chunks = []
        for chunk_start in range(start, end, max_size):
            chunk_end = min(chunk_start + max_size, end)
            if text[chunk_start:chunk_end].strip():
                chunks.append(MinimalSource(
                    file_path=file_path,
                    first_character_index=chunk_start,
                    last_character_index=chunk_end,
                ))
        return chunks
