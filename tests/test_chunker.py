"""Verify Python source coverage and exact character coordinates."""

import unittest

from src.chunker import Chunker


class PythonChunkerTests(unittest.TestCase):
    """Cover text formerly excluded by the top-level AST filter."""

    def check_coverage(self, text: str, size: int = 40) -> None:
        """Require every non-whitespace character exactly once, in order."""
        sources = Chunker.chunk_python("data/raw/sample.py", text, size)
        covered = [0] * len(text)
        previous_end = 0
        for source in sources:
            start = source.first_character_index
            end = source.last_character_index
            self.assertEqual(source.file_path, "data/raw/sample.py")
            self.assertGreaterEqual(start, previous_end)
            self.assertTrue(0 < end - start <= size)
            for i in range(start, end):
                covered[i] += 1
            previous_end = end
        for i, char in enumerate(text):
            if not char.isspace():
                self.assertEqual(covered[i], 1, (i, char))

    def test_globals_only(self) -> None:
        """Imports, module docstrings and constants all remain searchable."""
        self.check_coverage('"""Réglages."""\nimport os\nMAX_TOKENS = 2000\n')

    def test_blocks_and_gaps(self) -> None:
        """Include decorators, external comments and the main guard."""
        self.check_coverage(
            '# début\nimport os\n\n@decorator\ndef work():\n'
            '    """été"""\n    return 2\n\n# between\nSETTING = 3\n'
            'class Example:\n    value = "é"\n\n'
            'if __name__ == "__main__":\n    work()\n# end',
        )

    def test_decorated_function_keeps_original_span(self) -> None:
        """Small function blocks still include their decorators intact."""
        text = '# lead\n@decorator\nasync def work():\n    return "é"\n# end'
        sources = Chunker.chunk_python("test.py", text, 2000)
        passages = [text[s.first_character_index:s.last_character_index]
                    for s in sources]
        self.assertIn('@decorator\nasync def work():\n    return "é"\n',
                      passages)

    def test_invalid_syntax_falls_back_to_fixed_chunks(self) -> None:
        """Unparseable Python is still indexed without executing it."""
        self.check_coverage('def broken(:\n    value = "é"\n# comment\n', 12)

    def test_empty_and_invalid_size(self) -> None:
        """Empty text produces no chunks; nonpositive sizes are rejected."""
        self.assertEqual(Chunker.chunk_python("empty.py", "", 2000), [])
        with self.assertRaises(ValueError):
            Chunker.chunk_python("test.py", "import os", 0)


if __name__ == "__main__":
    unittest.main()
