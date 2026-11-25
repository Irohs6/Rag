import fire
from .rag_cli import RagCli
import sys


if __name__ == "__main__":
    try:
        fire.Fire(RagCli())
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
