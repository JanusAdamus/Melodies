PYTHON ?= python3
VENV ?= .venv
RUN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install test test-main test-next-token demo-score demo-main demo-transformer reproduce-demo clean clean-local-artifacts prune-redundant-artifacts

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

test:
	$(RUN) -m pytest tests next_token_experiment/tests

test-main:
	$(RUN) -m pytest tests

test-next-token:
	$(RUN) -m pytest next_token_experiment/tests

demo-score:
	$(RUN) examples/generate_example_score.py

demo-main: demo-score
	$(RUN) -m src.cli.main --input examples/example_score.musicxml --obs pitch_class --model both --output-dir artifacts/outputs/demo_compare

demo-transformer:
	$(RUN) -m next_token_experiment.cli --profile cpu_baseline --run-name cpu_baseline_smoke --max-files 8

reproduce-demo:
	$(RUN) scripts/reproduce_results.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov build dist .coverage

clean-local-artifacts:
	rm -rf artifacts/tmp artifacts/tmp_figs

prune-redundant-artifacts:
	rm -rf artifacts/next_token_experiment/results/cpu_baseline_smoke_v1
	rm -rf artifacts/next_token_experiment/results/cpu_baseline_smoke_v2
	rm -rf artifacts/next_token_experiment/results/cpu_baseline_smoke_v3
	rm -rf artifacts/next_token_experiment/results/cpu_baseline_smoke_v4
	rm -rf artifacts/next_token_experiment/results/cpu_baseline_smoke_v5
	rm -rf artifacts/next_token_experiment/results/library_smoke_8
	rm -rf artifacts/next_token_experiment/results/library_smoke_8_longer
	rm -rf artifacts/outputs/notebook_finite
	rm -rf artifacts/outputs/notebook_hdp
	rm -rf artifacts/outputs/classic_limited_eval
