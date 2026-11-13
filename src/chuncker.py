from .data_models import MinimalSource
import ast


def chunk_markdown(
    file_path: str,
    text: str,
    max_chunk_size,
) -> list[MinimalSource]:
    chunks = []
    for start in range(0, len(text, max_chunk_size)):
        end = min(start + max_chunk_size, len(text))
    chunks.append(
        MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end
        )
    )
    return chunks


def chunk_python(
    file_path: str,
    text: str,
    max_chunk_size,
) -> list[MinimalSource]:
    text = ast.parse(text)
    
