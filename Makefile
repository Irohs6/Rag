.PHONY: install run index search search_dataset answer answer_dataset \
	debug clean lint lint-strict test

ARGS ?= index
MAX_CHUNK_SIZE ?= 2000
K ?= 5
DATA_DIR ?= data/raw
INDEX_DIRECTORY ?= data/processed
DATASET_PATH ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
SAVE_DIRECTORY ?= data/output/search_results/UnansweredQuestions
STUDENT_SEARCH_RESULTS_PATH ?= data/output/search_results/UnansweredQuestions/dataset_docs_public.json
SAVE_A_DIRECTORY ?= data/output/search_results_and_answer
RESUME ?= False

install:
	uv sync

run:
	uv run python3 -m src $(ARGS)

index:
	uv run python3 -m src index \
		--data_dir "$(DATA_DIR)" \
		--max_chunk_size $(MAX_CHUNK_SIZE) \
		--save_directory "$(INDEX_DIRECTORY)"

search:
	uv run python3 -m src search "$(QUERY)" \
		--index_directory "$(INDEX_DIRECTORY)" --k $(K)

search_dataset:
	uv run python3 -m src search_dataset \
		--dataset_path "$(DATASET_PATH)" \
		--index_directory "$(INDEX_DIRECTORY)" \
		--k $(K) --save_directory "$(SAVE_DIRECTORY)"

answer:
	uv run python3 -m src answer "$(QUERY)" \
		--index_directory "$(INDEX_DIRECTORY)" --k $(K)

answer_dataset:
	uv run python3 -m src answer_dataset \
		--student_search_results_path "$(STUDENT_SEARCH_RESULTS_PATH)" \
		--save_directory "$(SAVE_A_DIRECTORY)" \
		--k $(K) --resume $(RESUME)

debug:
	uv run python3 -m pdb -m src $(ARGS)

clean:
	find src tests -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run python3 -m unittest discover -s tests -v
