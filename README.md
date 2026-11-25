*This project has been created as part of the 42 curriculum by gacattan.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation project for answering questions about the provided vLLM codebase. The intended pipeline indexes source files, retrieves relevant snippets, and generates grounded answers with Qwen/Qwen3-0.6B.

**Status: indexing, retrieval and CPU answer generation are implemented. The development `evaluate` command remains to be implemented.**

See [generation and resume behavior](docs/GENERATION.md) for the shared Answerer, context limits, checkpoints and Python chunk coverage.

## Instructions

Development targets Python 3.10+ and uv. Place the supplied corpus in `data/raw/` and the supplied question datasets in `data/datasets/`. See [resource placement](docs/RESOURCES.md) for the expected local paths. The corpus and datasets are intentionally excluded from Git.

Run commands from the repository root:

```sh
uv sync
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How is LoRA configured?" --k 5
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 5 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src answer "How is LoRA configured?" --k 5
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Run `make install` to synchronize the project environment. During development, `make run ARGS='index --data_dir data/raw'` builds and saves the index. See [the roadmap](docs/ROADMAP.md) and the assignment in [English](docs/subject_en.md) or [French](docs/subject_fr.md).

## System architecture

Implemented flow: corpus ingestion -> Python and text chunkers -> persistent lexical index -> ranked source spans -> context construction -> local Qwen generation -> validated JSON.

Use Pydantic models for exchanged data, Python Fire for the CLI, and tqdm for long-running operations. All input and output locations must be configurable.

## Chunking strategy

Python uses AST block boundaries plus the text between blocks; Markdown/text uses bounded slices near punctuation with word overlap. Preserve exact corpus paths and character offsets. Chunks must not exceed the configured maximum, which defaults to 2,000 characters and must never exceed the grader's 2,000-character source limit.

## Retrieval method

BM25S indexes the advanced tokenizer output for every document type. Search combines basic and advanced query tokens. Index arrays and ordered source coordinates are persisted separately.

## Performance analysis

Previously reported public-dataset Recall@5 was 85% for documentation and 61.6% for code. These measurements predate the expanded Python chunk coverage and must be rerun after rebuilding the index. Required targets from the subject:

- Recall@5: at least 80% for documentation and 50% for code.
- Full-corpus indexing: at most five minutes.
- Retrieval for 200 questions: at most 90 seconds.

## Design decisions

Keep generated indexes, question datasets, model weights, and outputs out of Git. Preserve the original corpus path prefix because evaluation compares paths verbatim. Keep the assignment PDF as a local reference, outside the published source license.

## Challenges faced

The implementation preserves original character spans, splits code identifiers for retrieval and shares token-budget handling between single and batch generation. Batch generation supports checkpoints and explicit resume. See [implementation details](docs/GENERATION.md).

## Resources

The project requirements were read from the supplied subject, version 2.0. Markdown adaptations are available in [English](docs/subject_en.md) and [French](docs/subject_fr.md). An optional local reference copy is `docs/subject.pdf` (ignored by Git). References: [BM25S](https://github.com/xhluca/bm25s), [Transformers](https://huggingface.co/docs/transformers/), [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Python AST](https://docs.python.org/3/library/ast.html), and [tqdm](https://tqdm.github.io/).

AI assistance was used to interpret the subject, initialize documentation, discuss and debug the RAG implementation, and implement the shared Answerer, context checks, resumable checkpoints, expanded Python chunk coverage and regression tests. Public-dataset performance measurements were run locally by the author; mocked generation tests do not measure model accuracy.

## License

Original project code is licensed under the [MIT License](LICENSE). The assignment, supplied datasets, external corpus, dependencies, and model weights retain their respective licenses.
