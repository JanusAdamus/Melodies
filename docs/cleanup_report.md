# Cleanup Report

## 1. What was changed

This cleanup focused on reproducibility, repository legibility, and safe
separation between active code, exploratory material, and generated outputs.

Main changes:

- added `src/project_paths.py` as a single helper for repository-relative paths;
- removed the hardcoded thesis path from
  `scripts/generate_thesis_chapter3_assets.py`;
- added `scripts/reproduce_results.py` as a minimal reproducibility entry point;
- moved demo notebooks to `notebooks/exploratory/`;
- cleared notebook outputs that embedded local absolute paths and stale runtime
  noise;
- updated installation and test flow to use `pytest`;
- added a `dev` extra in `pyproject.toml` and moved `jupyter` to a notebook-only
  extra;
- improved `README.md`, `docs/setup-and-reproduction.md`,
  `docs/project-structure.md`, `docs/experiments.md`, and `docs/transformer.md`
  to explain the actual current workflow;
- clarified artifact retention policy and canonical local references;
- added a small amount of structure around notebooks and scripts with local
  `README.md` files;
- renamed several default output directories in CLI entry points to more
  descriptive names;
- pruned clearly redundant local artifacts and temporary directories.

## 2. What was moved and why

### Notebooks

Moved:

- `notebooks/demo_finite_hmm.ipynb`
- `notebooks/demo_hdp_hmm.ipynb`
- `notebooks/demo_library_corpus.ipynb`
- `notebooks/demo_multicorpus.ipynb`

to:

- `notebooks/exploratory/`

Reason:

- these notebooks are demos and exploratory material, not the main interface of
  the repository;
- the move makes the top-level repo easier to read;
- the notebooks were also cleaned to avoid embedded absolute local paths.

### Outputs policy

No global migration from `artifacts/` to a new `outputs/` root was performed.

Reason:

- the repository was already consistently using `artifacts/` as the output
  root;
- changing that convention would risk breaking existing thesis references and
  historical runs without providing enough benefit.

## 3. What was deleted and why

Deleted local generated artifacts only when they were clearly redundant or
temporary:

- `artifacts/tmp`
- `artifacts/tmp_figs`
- `artifacts/next_token_experiment/results/cpu_baseline_smoke_v1`
- `artifacts/next_token_experiment/results/cpu_baseline_smoke_v2`
- `artifacts/next_token_experiment/results/cpu_baseline_smoke_v3`
- `artifacts/next_token_experiment/results/cpu_baseline_smoke_v4`
- `artifacts/next_token_experiment/results/cpu_baseline_smoke_v5`
- `artifacts/next_token_experiment/results/library_smoke_8`
- `artifacts/next_token_experiment/results/library_smoke_8_longer`
- `artifacts/outputs/notebook_finite`
- `artifacts/outputs/notebook_hdp`
- `artifacts/outputs/classic_limited_eval`

Reason:

- these were either temporary outputs, notebook-generated outputs, or older
  variants clearly superseded by newer retained runs.

No thesis-relevant raw corpora or primary source files were deleted.

## 4. What was not changed because it was risky

The following areas were intentionally left structurally close to their
previous state:

- scientific implementations in `src/models/`, `src/data/`, and
  `src/analysis/`;
- the package namespace `src` itself;
- the separate `next_token_experiment/` package layout;
- historical thesis/reference documents under `docs/reference/`;
- existing external corpora under `external/`;
- placeholder predictive wrappers in
  `next_token_experiment/models/finite_hmm.py` and
  `next_token_experiment/models/hdp_hmm.py`.

Reason:

- changing algorithms or moving package roots would create a real risk of
  altering thesis results or silently breaking historical workflows.

## 5. Remaining technical debt

- the final comparable evaluation `finite_hmm` vs `hdp_hmm` vs transformer
  under one shared `next-token` protocol is still incomplete;
- `next_token_experiment/models/finite_hmm.py` and `hdp_hmm.py` remain
  placeholders for predictive evaluation;
- several local artifact folders in `artifacts/outputs/` still have ambiguous
  names and should be reviewed one by one before final thesis freeze;
- the repository still mixes Spanish thesis-facing terminology with English code
  conventions, which is acceptable but not perfectly uniform;
- some historical docs remain long and redundant by design because they may
  still carry useful thesis context.

## 6. How to reproduce the current results

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

### Run tests

```bash
pytest tests next_token_experiment/tests
```

### Run a minimal reproducible demo

```bash
python scripts/reproduce_results.py
```

This will:

- generate the example score if needed;
- run the classical single-piece analysis;
- run a small `cpu_baseline` next-token demo if `external/library/scores`
  exists;
- otherwise skip the next-token part with a clear message.

### Run thesis-facing chapter 3 asset export

```bash
python scripts/generate_thesis_chapter3_assets.py
```

By default this writes to:

- `artifacts/thesis_export/`

You can override that with:

```bash
MELODIES_THESIS_ROOT=/path/to/thesis python scripts/generate_thesis_chapter3_assets.py
```

or:

```bash
python scripts/generate_thesis_chapter3_assets.py --thesis-root /path/to/thesis
```

## 7. Assumptions made

- `artifacts/` should remain the canonical local output root;
- notebooks are exploratory unless explicitly tied to the thesis manuscript;
- preserving scientific behavior is more important than forcing a new package
  hierarchy;
- local corpora in `external/` are intentionally outside the normal GitHub
  distribution path;
- references to thesis results should prefer documentation and reproducible
  scripts over large committed output bundles.
