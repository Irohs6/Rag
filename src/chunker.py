import ast

from .data_models import MinimalSource
import re


class Chunker:

    @staticmethod
    def chunk_markdown(
        file_path: str,
        text: str,
        max_chunk_size: int,
        overlap: int,
    ) -> list[MinimalSource]:
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
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size doit être supérieur à 0")

        parse_code = ast.parse(text)
        chunks = []

        # Position de départ de chaque ligne dans text.
        line_starts = [0]
        for line in text.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(line))

        for element in parse_code.body:
            if isinstance(
                element,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                first_line = element.lineno

                # Inclure les décorateurs dans le chunk.
                if element.decorator_list:
                    first_line = element.decorator_list[0].lineno

                last_line = element.end_lineno
                if last_line is None:
                    continue

                start = line_starts[first_line - 1]
                end = line_starts[last_line]

                # Découper davantage les blocs trop longs.
                for chunk_start in range(start, end, max_chunk_size):
                    chunk_end = min(chunk_start + max_chunk_size, end)

                    chunks.append(
                        MinimalSource(
                            file_path=file_path,
                            first_character_index=chunk_start,
                            last_character_index=chunk_end,
                        )
                    )
        return chunks
