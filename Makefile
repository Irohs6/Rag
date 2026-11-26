.PHONY: install run debug clean lint lint-strict

MAX_CHUNK_SIZE = 2000
K = 5
DATASET_PATH = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
SAVE_DIRECTORY = "data/output/search_results/UnansweredQuestions"
STUDENT_SEARCH_RESULTS_PATH = "data/output/search_results/UnansweredQuestions/dataset_docs_public.json"
SAVE_A_DIRECTORY = "data/output/search_results_and_answer/"

install:
	uv sync

index:
	uv run python3 -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

search:
	uv run python3 -m src search "$(QUERY)" --k $(K)

search_dataset:
	uv run python3 -m src search_dataset --dataset_path "$(DATASET_PATH)" --k $(K) --save_directory "$(SAVE_DIRECTORY)"

answer:
	uv run python3 -m src answer "$(QUERY)" --k $(K)

answer_dataset:
	uv run python3 -m src answer_dataset  --student_search_results_path "$(STUDENT_SEARCH_RESULTS_PATH)" --save_directory "$(SAVE_A_DIRECTORY)"

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find src -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
