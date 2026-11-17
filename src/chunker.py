import ast

from .data_models import MinimalSource


class Chunker:

    @staticmethod
    def chunk_markdown(
        file_path: str,
        text: str,
        max_chunk_size: int,
    ) -> list[MinimalSource]:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size doit être supérieur à 0")

        chunks = []

        for start in range(0, len(text), max_chunk_size):
            end = min(start + max_chunk_size, len(text))

            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                )
            )

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
