

def chunk_markdown(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000,
) -> list[MinimalSource]: