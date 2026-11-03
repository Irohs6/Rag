# Implementation roadmap

Planning checklist based on subject version 2.0; this is not a completion report.

## 1. Development setup

- [x] Initialize repository structure and MIT license.
- [ ] Confirm the author's 42 login and update the README first line.
- [ ] Add pyproject.toml and generate uv.lock; verify uv sync.
- [ ] Configure Python 3.10+, Pydantic, Fire, tqdm, flake8, and mypy.
- [ ] Add Makefile rules: install, run, debug, clean, lint.
- [ ] Run flake8 . and mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs.
- [ ] Obtain the supplied vLLM corpus and reference datasets.

## 2. Data contracts and CLI

- [ ] Implement MinimalSource, UnansweredQuestion, AnsweredQuestion, RagDataset, MinimalSearchResults, MinimalAnswer, StudentSearchResults, and StudentSearchResultsAndAnswer.
- [ ] Expose index, search, search_dataset, answer, answer_dataset, and evaluate through uv run python -m src.
- [ ] Make all input and output paths configurable.
- [ ] Handle empty queries, nonsensical queries, k=0, missing files, malformed JSON, and invalid options without unhandled tracebacks.

## 3. Indexing and retrieval

- [ ] Implement distinct Python and Markdown/text chunkers.
- [ ] Preserve exact corpus paths, text, and character spans.
- [ ] Enforce the configured size with a 2,000-character ceiling.
- [ ] Persist a BM25 or TF-IDF index under data/processed/.
- [ ] Retrieve ranked top-k source spans for individual and batch queries.
- [ ] Write StudentSearchResults JSON; keep output directories scoped by dataset.

## 4. Grounded generation

- [ ] Load Qwen/Qwen3-0.6B locally.
- [ ] Construct context within the token budget and ground answers in the retrieved evidence.
- [ ] Validate output with Pydantic and write StudentSearchResultsAndAnswer JSON.

## 5. Evaluation and documentation

- [ ] Implement local recall@k using same-file character-span overlap (IoU threshold 0.05).
- [ ] Never import or invoke the moulinette from the application.
- [ ] Reach recall@5 >= 0.80 on docs and >= 0.50 on code.
- [ ] Measure indexing <= 5 minutes and 200-query retrieval <= 90 seconds.
- [ ] Verify edge cases, type hints, docstrings, resource cleanup, and error handling.
- [ ] Record measured performance, chunk-size effects, decisions, difficulties, examples, references, and AI usage in the English README.

## Optional bonuses after mandatory validation

- [ ] CPU semantic embeddings.
- [ ] Hybrid lexical and semantic ranking.
- [ ] Incremental indexing.
- [ ] Index and query caching.
- [ ] Local HTTP API.
