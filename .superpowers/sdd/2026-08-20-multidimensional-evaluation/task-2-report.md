# Task 2 Report: VOMM Bridge Control

## Status

Complete. Task 2 was implemented only in the assigned worktree on
`codex/multidimensional-evaluation`, starting from `324ee13`.

## Changed files

- `Comparacion/vomm.py` — interpolated variable-order Markov diagnostic.
- `Comparacion/__init__.py` — exports the VOMM public interface.
- `tests/test_vomm.py` — literal, hand-derived behavioral tests.

## RED evidence

Command:

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_vomm -v
```

Observed failure before production code existed: `ModuleNotFoundError: No module named 'Comparacion.vomm'`.

## GREEN evidence

Focused command:

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_vomm -v
```

Result: 5 tests run, all passed.

Full documented suites:

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s next_token_experiment/tests -v
```

Results: 31 tests run, all passed; 19 tests run, all passed.

## Probability and backoff design

`VariableOrderMarkovModel` is a small PPM-inspired diagnostic control, explicitly not an IDyOM implementation. It counts next-token observations for every order from zero through `max_order`, adding `vocabulary_size - 1` as BOS independently to every training piece. Evaluation uses `build_evaluation_slices`, begins every slice from BOS, and scores every musical token exactly once.

Prediction begins with the alpha-smoothed unigram posterior. Each observed suffix context then blends its alpha-smoothed local posterior with the current lower-order distribution using `context_count / (context_count + backoff_strength)`. Missing contexts are skipped, so prediction backs off to the available shorter suffix. The result is normalized defensively, with a uniform fallback if numerical input is invalid.

Validation selection fits each candidate order independently, scores validation NLL, and returns the retained model that produced the lowest value. Evaluation exposes the standard model, selected-order, validation/test NLL, perplexity, token-count, count-table/parameter-size, fit-time, evaluation-time, and per-piece metrics fields.

## Commit

Implementation commit: `fb5a0c3 feat: add variable-order Markov bridge baseline`.

## Concerns

The control assumes the existing primary-token convention that BOS is `vocabulary_size - 1`; callers with another BOS allocation must remap tokens before fitting. Runner integration, canonical training/corpus scans, and the deferred FFBS findings were intentionally left for later tasks.
