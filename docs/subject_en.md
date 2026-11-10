# RAG against the machine

*Will you answer my questions?*  
**Subject version: 2.0**

> Markdown adaptation of the supplied English assignment. The supplied PDF remains the reference if a formatting or transcription difference is found.

## Contents

1. [Preamble](#i-preamble)
2. [Foreword](#ii-foreword)
3. [AI instructions](#iii-ai-instructions)
4. [Introduction](#iv-introduction)
5. [Common instructions](#v-common-instructions)
6. [Mandatory part](#vi-mandatory-part)
7. [Evaluation](#vii-evaluation)
8. [README requirements](#viii-readme-requirements)
9. [Bonus part](#ix-bonus-part)
10. [Submission and peer evaluation](#x-submission-and-peer-evaluation)

## I. Preamble

Build a pipeline that ingests and indexes a codebase, retrieves relevant snippets for a question, passes them to a small local model to generate a grounded answer, and measures retrieval quality with recall@k.

## II. Foreword

A language model's knowledge is fixed at the end of training. It may not know a recent library, a private codebase, or a new incident, and retraining it for every change is costly. Retrieval augmented generation (RAG) supplies external information at answer time. In this project, that information is a large codebase. The engineering challenge is to identify a few passages that answer a question without omitting the evidence or overwhelming the small model with irrelevant text.

## III. AI instructions

AI tools can help with code, documentation, ideas, and explanations, but their output may be wrong or biased. Understand and verify anything you use, and seek peer review.

The learning goals are to use AI to reduce repetitive work, develop prompting skills, understand its risks, and improve problem solving and collaboration. Think through the problem before prompting, use precise vocabulary, and systematically test and question generated work. You must be able to explain and take responsibility for every part of the project.

Good examples include asking AI for test ideas or a parser design, then checking and refining the result with peers. Copying a complete generated function that you cannot explain during peer evaluation is a bad example.

## IV. Introduction

After function calling in `call_me_maybe`, this project explores RAG. Instead of retraining the model to add knowledge, give it access to an external source at answer time. The four stages are:

1. **Indexing:** organize the data for search.
2. **Retrieving:** match a question to the index and select relevant snippets.
3. **Augmenting:** filter snippets and place them in the model's context window.
4. **Generating:** produce an answer using that context.

Read the entire subject before coding because each stage feeds the next.

## V. Common instructions

### V.1 General rules

- Use Python 3.10 or later and follow the `flake8` standard.
- Handle exceptions gracefully. An unhandled crash during review makes the program nonfunctional. Prefer context managers and manage resources without leaks.
- Add type hints to function parameters, return values, and variables where applicable. All functions must pass `mypy`.
- Write PEP 257 docstrings for functions and classes, including their purpose, parameters, and return values where relevant.

### V.2 Makefile

The mandatory rules are:

| Rule | Action |
| --- | --- |
| `install` | Install the dependencies with the chosen package manager. |
| `run` | Run the main script. |
| `debug` | Run the script in a debugger such as `pdb`. |
| `clean` | Remove temporary files and caches. |
| `lint` | Run `flake8 .` and the specified `mypy` command. |

The required `mypy` command is:

```sh
mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
```

An optional `lint-strict` rule runs `flake8 .` and `mypy . --strict`.

### V.3 Additional guidelines

- Write tests, for example with `pytest` or `unittest`, including edge cases. Test programs are not submitted or graded.
- Include a `.gitignore` for Python artifacts.
- A virtual environment is recommended for development.

### V.4 Project-specific requirements

- Use **Pydantic** to validate the data models exchanged between stages. Service and orchestration classes such as indexers, retrievers, and pipelines need not inherit from Pydantic models.
- `flake8` also applies to bonus files.
- Use **uv** to manage the project and its packages. The reviewer and the moulinette run `uv sync`.
- Expose the CLI with **Python Fire** and use **tqdm** progress bars for long operations.
- Use **Qwen/Qwen3-0.6B** by default. Other models are allowed only if the project continues to work with Qwen/Qwen3-0.6B.
- Beyond these required tools, any library is allowed.

## VI. Mandatory part

Build a RAG system that answers questions about the supplied **vLLM repository**. Index its files, retrieve relevant passages, generate answers with Qwen/Qwen3-0.6B, and assess retrieval with recall@k. Grading focuses on correct source locations and grounded answers.

### VI.1 Indexing the codebase

Read the files you consider useful from the supplied vLLM repository, split them into chunks, and persist an index under `data/processed/`. Indexing the whole corpus must take **at most five minutes**.

Implement **two distinct chunking strategies**: one for Python code, another for Markdown and text. Implement at least one classic **lexical retrieval** method: **TF-IDF or BM25**. Other methods may be added on top.

The `--max_chunk_size` CLI option controls chunk size and defaults to **2,000 characters**. Never return a source longer than 2,000 characters: the moulinette's `max_context_length` is 2,000, and one oversized source invalidates the entire output. Smaller chunks are permitted; report their impact on recall@k.

What you preserve at indexing time matters: a question may paraphrase the source or quote an identifier exactly.

### VI.2 Retrieval

Return the top `k` ranked source locations for a question. Every result has a `file_path`, `first_character_index`, and `last_character_index`, with a span no wider than 2,000 characters.

The path must match the corpus path **exactly**, including its prefix, for example `data/raw/vllm-0.10.1/docs/features/lora.md`. The grader compares paths verbatim.

Support both a single query and batch search over a JSON dataset. Reach **recall@5 ≥ 80%** on documentation questions and **recall@5 ≥ 50%** on code questions in the reference datasets.

### VI.3 Answer generation

Pass the retrieved snippets to Qwen/Qwen3-0.6B within its token budget. Produce a natural-language answer and structured JSON conforming to the provided Pydantic models.

Answers should be understandable, relevant to the question, and grounded in the sources without major hallucinations. Because the small model has reasoning limits, perfect phrasing is not a strict pass/fail requirement. Grading prioritizes retrieval quality, grounding, and the prompt strategy. A coherent, mostly grounded but partly incomplete answer can be valid.

### VI.4 Data models

Implement these Pydantic models. The types and field names below are the subject's minimum contract:

```python
import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
```

You may add models and fields if needed, while preserving the required minimum structure.

### VI.5 Output

Search output conforms to `StudentSearchResults`; answer output conforms to `StudentSearchResultsAndAnswer`. The `file_path` is relative to the project root and exactly matches the ingested corpus path. Character indices identify the beginning and end of the source in that file.

Example search output:

```json
{
  "search_results": [
    {
      "question_id": "q1",
      "question": "How to configure OpenAI server?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
          "first_character_index": 9867,
          "last_character_index": 10100
        },
        {
          "file_path": "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
          "first_character_index": 267,
          "last_character_index": 400
        }
      ]
    }
  ],
  "k": 10
}
```

The answer format adds an `answer` string to each item in `search_results`, for example `"answer": "To configure the OpenAI compatible server in vLLM..."`.

### VI.6 Command-line interface

Use Python Fire. Invoke every command as `uv run python -m src <command> [options]`. These commands and minimum options are required:

| Command | Purpose |
| --- | --- |
| `index --max_chunk_size <int>` | Ingest `data/raw/` and build an index in `data/processed/`. |
| `search <query> --k <int>` | Retrieve the top `k` sources for one query. |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir>` | Search every question in a JSON dataset and save `StudentSearchResults`. |
| `answer <query> --k <int>` | Answer one query using retrieved context. |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | Generate answers from saved search results and save `StudentSearchResultsAndAnswer`. |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | Measure recall@k against ground truth for your own testing. |

All input and output paths must be configurable CLI arguments rather than hard-coded. Add progress bars with `tqdm` for long operations. Handle empty or nonsensical queries, `k=0`, missing files, and malformed JSON gracefully, without an unhandled traceback.

The `evaluate` command is for development. The official recall@k comes from the provided moulinette executable. Your application must **never import or invoke the moulinette**.

### VI.7 End-to-end walkthrough

The defense runs an automated pipeline: `index` → `search_dataset` → moulinette `evaluate_student_search_results` → `answer_dataset`. The single-query commands remain available for debugging.

The expected layout during evaluation is:

```text
src/
pyproject.toml
uv.lock
README.md
data/
  raw/                                  # supplied vLLM repository
  processed/                            # generated index
  datasets/
    UnansweredQuestions/
    AnsweredQuestions/
  output/
    search_results/<DatasetScope>/
    search_results_and_answer/<DatasetScope>/
```

The scripts supply their own path arguments and output directories. A wrong interface or structure prevents the reference scripts from running and causes the corresponding checks to fail.

Illustrative commands (run each on one shell line, or use shell line continuation):

```sh
uv run python -m src index --max_chunk_size 2000

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Rename the supplied platform-specific executable (`moulinette-ubuntu` or `moulinette-fedora`) to `moulinette` before the example evaluation command. The student output is its first positional argument and the answered ground-truth dataset is the second. The example scores printed in the PDF illustrate output formatting, **not** target scores.

Scope `--save_directory` by `UnansweredQuestions` or `AnsweredQuestions`: public datasets may have the same filenames, so reusing one output directory can overwrite results.

## VII. Evaluation

### VII.1 Recall@k

For each question, recall@k is the share of its correct sources found among the top `k` results. A source counts as found if a retrieved span is in the **same file** and overlaps the reference character span. The intersection-over-union (IoU) threshold is **0.05**. Exact span boundaries are unnecessary; the right region in the right file is sufficient.

The moulinette validates the output and reports recall at several values of `k`.

### VII.2 Performance thresholds

| Measure | Required result |
| --- | ---: |
| Whole-corpus indexing | ≤ 5 minutes |
| Retrieval for 200 questions | ≤ 90 seconds |
| Documentation recall@5 | ≥ 80% |
| Code recall@5 | ≥ 50% |

## VIII. README requirements

The root `README.md` must be written in English. Its **first line must be italicized** and read:

> *This project has been created as part of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]].*

It needs a Description, Instructions, and Resources section. Resources must list conventional references and describe how AI was used, including the tasks and parts of the project involved. It must also explain:

- System architecture and the relationships between pipeline components.
- Chunking strategy.
- Retrieval algorithm and ranking.
- Recall@k and performance measurements.
- Design decisions and challenges.
- Example commands.

## IX. Bonus part

Bonus work is evaluated **only after the entire mandatory part passes**, including robustness to unusual reviewer inputs. Each of five working bonuses earns one point. All must run on a **CPU-only campus machine**; documentation alone does not count.

1. **Semantic embeddings:** a vector index beside the lexical index, using a lightweight CPU model such as `all-MiniLM-L6-v2`.
2. **Hybrid retrieval:** combine lexical and semantic rankings.
3. **Incremental indexing:** re-index only a changed file.
4. **Caching:** cache the index and query results to accelerate cold starts and repeated queries.
5. **Local HTTP API:** allow another client to search and answer without the CLI.

## X. Submission and peer evaluation

Submit the Git repository. Only repository contents are reviewed. Include:

- `src/` with the implementation.
- `pyproject.toml` and `uv.lock`.
- A `Makefile` with `install`, `run`, `debug`, `clean`, and `lint`.
- A comprehensive `README.md` and any necessary configuration.

**Do not commit large data files, model weights, or generated outputs.** The evaluator supplies or generates them. The model and deep-learning stack can require several gigabytes; develop in a location with sufficient disk space.

During peer evaluation you may be asked to make a small code, behavior, display, or data-structure change in a few minutes. Be ready to explain and modify your own work.
