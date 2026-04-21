PYTHON ?= python3
VENV ?= .venv
RUN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install test test-main test-next-token demo-score demo-main demo-transformer clean

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .

test: test-main test-next-token

test-main:
	$(RUN) -m unittest discover -s tests

test-next-token:
	$(RUN) -m unittest discover -s next_token_experiment/tests

demo-score:
	$(RUN) examples/generate_example_score.py

demo-main: demo-score
	$(RUN) -m src.cli.main --input examples/example_score.musicxml --obs pitch_class --model both --output-dir artifacts/outputs/demo_compare

demo-transformer:
	$(RUN) -m next_token_experiment.cli --profile cpu_baseline --run-name cpu_baseline_smoke --max-files 8

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov build dist .coverage
