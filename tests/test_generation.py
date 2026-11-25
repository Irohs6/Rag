"""Regression tests for generation, context limits and resumable output."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import Mock, patch

from src.answerer import Answerer
from src.answer_progress import AnswerProgress, write_json_atomic
from src.data_models import MinimalSearchResults, MinimalSource
from src.rag_cli import RagCli


class GenerationTests(unittest.TestCase):
    """Exercise generation orchestration without downloading model weights."""

    def setUp(self) -> None:
        """Create source files and replace only the Transformers backend."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.file = self.root / "source.md"
        self.file.write_text(
            "first evidence\nsecond evidence", encoding="utf-8",
        )
        self.sources = [
            MinimalSource(file_path=str(self.file),
                          first_character_index=0, last_character_index=14),
            MinimalSource(file_path=str(self.file),
                          first_character_index=15, last_character_index=30),
        ]
        self.result = MinimalSearchResults(
            question_id="q1", question="What evidence?",
            retrieved_sources=self.sources,
        )
        self.tokenizer = Mock()
        self.tokenizer.apply_chat_template.side_effect = self.format_prompt
        self.tokenizer.side_effect = self.encode
        self.model = Mock()
        self.model.config.max_position_embeddings = 10000
        self.pipe = Mock(return_value=[{"generated_text": " An answer "}])
        self.model_loader = self.mock_backend(
            "AutoModelForCausalLM.from_pretrained", self.model,
        )
        self.mock_backend("AutoTokenizer.from_pretrained", self.tokenizer)
        self.mock_backend("pipeline", self.pipe)

    def mock_backend(self, name: str, value: Any) -> Mock:
        """Patch one Transformers constructor and arrange automatic cleanup."""
        replacement = patch(f"src.answerer.{name}", return_value=value)
        mock = replacement.start()
        self.addCleanup(replacement.stop)
        return mock

    @staticmethod
    def format_prompt(messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Represent chat framing explicitly in the fake token budget."""
        return "<chat>" + json.dumps(messages) + "<assistant>"

    @staticmethod
    def encode(prompt: str, **kwargs: Any) -> dict[str, list[int]]:
        """Use one fake token per character to test exact budget boundaries."""
        return {"input_ids": [1] * len(prompt)}

    def write_input(self) -> Path:
        """Write two questions with two retrieved sources each."""
        path = self.root / "questions.json"
        second = self.result.model_copy(update={"question_id": "q2"})
        path.write_text(json.dumps({
            "search_results": [self.result.model_dump(), second.model_dump()],
            "k": 2,
        }), encoding="utf-8")
        return path

    def test_model_reused_and_k_applied(self) -> None:
        """Share a model and supply only the selected source."""
        answerer = Answerer()
        for _ in range(2):
            answer = answerer.answer(self.result, k=1)
            self.assertEqual(answer.retrieved_sources, self.sources[:1])
            self.assertEqual(answer.answer, "An answer")
            self.assertEqual(answer.question_id, "q1")
        self.assertEqual(self.model_loader.call_count, 1)
        self.assertNotIn("second evidence", self.pipe.call_args.args[0])
        self.assertFalse(self.pipe.call_args.kwargs["do_sample"])
        template_args = self.tokenizer.apply_chat_template.call_args.kwargs
        self.assertFalse(template_args["enable_thinking"])
        self.assertTrue(template_args["add_generation_prompt"])

    def test_no_sources_does_not_load_model(self) -> None:
        """An empty retrieval produces an explicit answer without inference."""
        result = self.result.model_copy(update={"retrieved_sources": []})
        answer = Answerer().answer(result)
        self.assertIn("No sources", answer.answer)
        self.model_loader.assert_not_called()

    def test_context_budget_drops_low_ranked_sources(self) -> None:
        """Reserve output tokens and retain complete, highest-ranked spans."""
        answerer = Answerer()
        answerer.answer(self.result, k=1)
        first_prompt = self.pipe.call_args.args[0]
        answerer.context_limit = len(first_prompt) + Answerer.MAX_NEW_TOKENS
        answer = answerer.answer(self.result, k=2)
        self.assertEqual(answer.retrieved_sources, self.sources[:1])
        self.assertEqual(self.pipe.call_args.args[0], first_prompt)
        self.pipe.reset_mock()
        answerer.context_limit -= 1
        with self.assertRaisesRegex(ValueError, "context budget"):
            answerer.answer(self.result, k=2)
        self.pipe.assert_not_called()

    def test_invalid_ranges_and_k_rejected(self) -> None:
        """Invalid inputs cannot silently generate from a truncated slice."""
        with self.assertRaises(ValueError):
            Answerer().answer(self.result, k=0)
        bad = self.sources[0].model_copy(update={"last_character_index": 999})
        result = self.result.model_copy(update={"retrieved_sources": [bad]})
        with self.assertRaisesRegex(ValueError, "source range"):
            Answerer().answer(result)
        self.pipe.assert_not_called()

    def test_single_command_uses_shared_answerer(self) -> None:
        """The single-question CLI keeps its string return contract."""
        with patch.object(RagCli, "search", return_value=self.result):
            self.assertEqual(RagCli.answer("What evidence?", k=1), "An answer")
        self.assertNotIn("second evidence", self.pipe.call_args.args[0])

    def test_interruption_resume_and_complete_output(self) -> None:
        """A failed second answer resumes without regenerating the first."""
        input_file = self.write_input()
        output_dir = self.root / "answers"
        self.pipe.side_effect = [
            [{"generated_text": "first answer"}], RuntimeError("interrupted"),
        ]
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            RagCli.answer_dataset(str(input_file), str(output_dir), k=1)
        checkpoint = output_dir / "questions.progress.json"
        progress = AnswerProgress.model_validate_json(checkpoint.read_text())
        self.assertEqual(len(progress.answers.search_results), 1)
        self.assertFalse((output_dir / "questions.json").exists())
        self.pipe.reset_mock(side_effect=True)
        RagCli.answer_dataset(
            str(input_file), str(output_dir), k=1, resume=True,
        )
        self.assertEqual(self.pipe.call_count, 1)
        data = json.loads((output_dir / "questions.json").read_text())
        self.assertEqual(data["k"], 1)
        self.assertEqual([r["question_id"] for r in data["search_results"]],
                         ["q1", "q2"])
        self.assertEqual(data["search_results"][0]["answer"], "first answer")
        self.assertTrue(all(len(r["retrieved_sources"]) == 1
                            for r in data["search_results"]))
        self.model_loader.reset_mock()
        RagCli.answer_dataset(
            str(input_file), str(output_dir), k=1, resume=True,
        )
        self.model_loader.assert_not_called()

    def test_resume_rejects_changed_settings_or_source(self) -> None:
        """Saved answers are never reused across different source contents."""
        input_file = self.write_input()
        output_dir = self.root / "answers"
        RagCli.answer_dataset(str(input_file), str(output_dir), k=1)
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            RagCli.answer_dataset(str(input_file), str(output_dir), k=2,
                                  resume=True)
        self.file.write_text("different contents", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            RagCli.answer_dataset(str(input_file), str(output_dir), k=1,
                                  resume=True)

    def test_failed_atomic_write_preserves_previous_json(self) -> None:
        """A failed replacement leaves the previous valid checkpoint intact."""
        path = self.root / "state.json"
        path.write_text('{"old": true}', encoding="utf-8")
        with patch.object(Path, "replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                write_json_atomic(path, '{"new": true}')
        self.assertEqual(json.loads(path.read_text()), {"old": True})
        self.assertEqual(list(self.root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
