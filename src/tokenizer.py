import re
from typing import cast

import bm25s


class Tokenizer:
    """Use advanced tokens for all documents and combined query tokens."""

    KEYWORDS_PYTHON = {
        "def", "class", "import", "from", "return",
        "if", "else", "elif", "for", "while",
        "try", "except", "with", "as", "pass",
        "none", "true", "false", "and", "or", "not",
        "in", "is", "lambda", "yield", "async",
        "await", "self",
    }

    @staticmethod
    def tokenize_advanced(
        texte: str,
        exclure_keywords: set[str] = KEYWORDS_PYTHON,
        min_len: int = 3
    ) -> list[str]:
        """
        Tokenize any corpus document, splitting code identifiers as well.
        """
        # Split camelCase and PascalCase words into separate tokens.
        texte = re.sub(r'([a-z])([A-Z])', r'\1 \2', texte)
        texte = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', texte)

        # Replace common separators with spaces.
        texte = re.sub(r'[_\-./\\:@#()\[\]{}<>,;=!?|&*+%^~`"\']+', ' ', texte)

        # Tokeniser
        tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', texte)
        tokens = [t.lower() for t in tokens]

        # Filtrer
        return [
            t for t in tokens
            if len(t) >= min_len and t not in exclure_keywords
        ]

    @staticmethod
    def tokenize_text(text: str) -> list[str]:
        """Use the basic BM25S tokenizer for text."""
        return cast(list[list[str]], bm25s.tokenize(
            text,
            return_ids=False,
            show_progress=False,
        ))[0]

    @staticmethod
    def tokenize_query(query: str) -> list[str]:
        """Tokenize a query using both text and Python tokenizers,
            removing duplicates."""
        tokens_base = Tokenizer.tokenize_text(query)
        tokens_code = Tokenizer.tokenize_advanced(query)

        return list(dict.fromkeys(tokens_base + tokens_code))
