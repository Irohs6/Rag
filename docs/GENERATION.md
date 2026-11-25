# Answer generation and resumable datasets

## Shared Answerer

`RagCli.answer` retrieves sources and calls `Answerer.answer` once. It still
returns plain text so Fire displays the answer in the terminal.
`RagCli.answer_dataset` creates one Answerer and reuses it for all questions.
The CLI commands remain static methods; the Answerer instance owns the loaded
model, tokenizer and generation pipeline.

Qwen/Qwen3-0.6B is loaded lazily, on the first question with sources. Empty
retrieval results and an already completed resumed run do not load the model.
Generation remains CPU-only, non-thinking and greedy, with at most 128 new tokens.

## Source selection and context budget

Both commands use the same steps:

1. Select at most `k` sources in retrieval order (`k=5` by default).
2. Read the original UTF-8 text and slice `[first_character_index:last_character_index]`.
3. Build the chat prompt, including source labels and the question.
4. Count the tokens of the complete formatted prompt and reserve 128 output tokens.
5. If necessary, remove the lowest-ranked source and repeat the check.
6. Generate an answer and return a `MinimalAnswer` with only the sources supplied.

Sources are never silently cut in half. If even the first source and question
cannot fit, generation raises a clear error. Shorten the question or rebuild
with smaller chunks. Missing files and invalid source coordinates also raise
errors rather than generating from incorrect evidence.

The dataset output's `k` is the requested generation limit; individual answers
may have fewer sources after context fitting. The input search-results JSON is
not modified, and should still be used for retrieval evaluation.

## Progressive saving and resume

```sh
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer \
  --k 5
```

After each completed answer, the command saves
`dataset_docs_public.progress.json` in the output directory. It contains the
partial answers and a fingerprint of the input questions, source coordinates,
selected source files' contents, source limit, model name and prompt settings.

Writes use a temporary file followed by an atomic replacement. A failed write
before replacement leaves the previous checkpoint intact. A question interrupted
while it is being generated is recalculated on resume.

To resume, rerun the command with the same arguments and `--resume True`:

```sh
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer \
  --k 5 \
  --resume True
```

Resume rejects missing/corrupt checkpoints or changed inputs/settings. Without
`--resume`, the command starts a new run and replaces the old checkpoint.
Use a different output directory to retain separate experiments. Do not run two
writers concurrently against the same output/checkpoint paths.

Only when all questions are complete is `dataset_docs_public.json` written in
the standard `StudentSearchResultsAndAnswer` format. The checkpoint is retained;
resuming a completed run simply re-exports the results. If an older complete
output exists during a new run, it stays unchanged until that new run completes.
The `.progress.json` file is internal and must not be submitted as final output.

## Python chunk coverage

Functions, async functions and classes keep their existing AST-based boundaries,
including decorators. The previously skipped gaps are now also chunked:
imports, constants, module docstrings, external comments and top-level control
flow. Files containing only these elements are therefore searchable.

Large regions still use bounded character slices. Whitespace-only slices are
ignored. If the runtime cannot parse a Python file, fixed-size slicing preserves
its contents without executing the file. Markdown chunking is unchanged.

Rebuild the index after this update, then regenerate both search datasets and
rerun the moulinette. More indexed content changes BM25 statistics and rankings;
previous recall measurements cannot be assumed to hold.

## Tokenization

`Tokenizer.tokenize_advanced` is used for every indexed document, including
Markdown and plain text. It is the existing tokenizer, renamed to reflect its
actual scope. The unused extension-based dispatcher was removed. Queries still
combine the basic and advanced tokens, deduplicating them in order.

## Validation

```sh
make lint
uv run python -m unittest discover -s tests -v
```

Regression tests cover exact source coverage, decorators, Unicode offsets,
syntax fallback, model reuse, source limits, context boundaries, interrupted
runs, checkpoint validation and atomic writes. Generation tests replace the
Transformers backend; they validate orchestration, not model quality or speed.
