*This project has been created as part of the 42 curriculum by <42_login_to_complete>.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation project for answering questions about the provided vLLM codebase. The intended pipeline indexes source files, retrieves relevant snippets, and generates grounded answers with Qwen/Qwen3-0.6B.

**Status: repository initialized. The RAG pipeline is not implemented yet.**

This starter contains the project structure, MIT license, and a roadmap based on subject version 2.0. Replace the placeholder above with the author's actual 42 login before submission.

## Instructions

Development targets Python 3.10+ and uv. Place the supplied corpus in `data/raw/` and the supplied question datasets in `data/datasets/`. See [resource placement](docs/RESOURCES.md) for the expected local paths. The corpus and datasets are intentionally excluded from Git.

The following interface must be implemented before these commands can run:

```sh
uv sync
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How is LoRA configured?" --k 5
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 5 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src answer "How is LoRA configured?" --k 5
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Run `make install` to synchronize the project environment. During development, `make run ARGS='index --data_dir data/raw'` lists the selected files. See [the roadmap](docs/ROADMAP.md) and the assignment in [English](docs/subject_en.md) or [French](docs/subject_fr.md).

## System architecture

Planned flow: corpus ingestion -> Python and text chunkers -> persistent lexical index -> ranked source spans -> context construction -> local Qwen generation -> validated JSON.

Use Pydantic models for exchanged data, Python Fire for the CLI, and tqdm for long-running operations. All input and output locations must be configurable.

## Chunking strategy

Implement separate strategies for Python and Markdown/text. Preserve exact corpus paths and character offsets. Chunks must not exceed the configured maximum, which defaults to 2,000 characters and must never exceed the grader's 2,000-character source limit.

## Retrieval method

Implement at least BM25 or TF-IDF. The final choice and ranking details remain to be measured and documented.

## Performance analysis

No benchmarks have been run. Required targets from the subject:

- Recall@5: at least 80% for documentation and 50% for code.
- Full-corpus indexing: at most five minutes.
- Retrieval for 200 questions: at most 90 seconds.

## Design decisions

Keep generated indexes, question datasets, model weights, and outputs out of Git. Preserve the original corpus path prefix because evaluation compares paths verbatim. Keep the assignment PDF as a local reference, outside the published source license.

## Challenges faced

Implementation has not started. Upcoming work includes preserving character spans, matching identifiers and paraphrases, and fitting retrieved evidence into the local model's token budget.

## Resources

The project requirements were read from the supplied subject, version 2.0. Markdown adaptations are available in [English](docs/subject_en.md) and [French](docs/subject_fr.md). An optional local reference copy is `docs/subject.pdf` (ignored by Git). Technical references will be added with the implementation.

AI assistance was used to read the subject and initialize the repository, directory structure, and planning documentation. No RAG implementation or performance results were generated in this setup step.

## License

Original project code is licensed under the [MIT License](LICENSE). The assignment, supplied datasets, external corpus, dependencies, and model weights retain their respective licenses.
