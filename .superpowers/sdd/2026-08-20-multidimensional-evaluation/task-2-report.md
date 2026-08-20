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

## Fix round 1/5

### Status and scope

Complete. Reviewer findings were addressed in `Comparacion/vomm.py` and `tests/test_vomm.py`. Runner, config, design docs, and integration code were not changed.

### RED evidence

Command:

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_vomm -v
```

After correcting the test harness to express missing fields as assertions, the clean RED run executed 11 tests and reported 22 expected assertion failures. Failures covered lossy constructor/token casts, non-finite smoothing, BOS targets and placement, absent accuracy/Brier fields, vacuous validation selection, and absent total-selection timing.

### Implemented corrections

- Constructor integer fields now require genuine non-boolean integral values; smoothing fields require finite positive real values.
- Training and evaluation reject non-integral musical targets and reject the explicit `bos_token_id = vocabulary_size - 1` as a target. Prediction validates token range and permits BOS only once at context position zero.
- Evaluation accumulates NLL, accuracy, and multiclass Brier score from each full distribution, both per piece and over all scored tokens.
- The asymmetric five-token coverage test instruments literal contexts across a three-token slice plus a two-token tail, detecting duplicated slice heads and omitted tails.
- Validation selection rejects zero total validation tokens and retains the actual lowest-NLL model/order.
- Selection timing wraps all candidate fits and validation evaluations. The comparable `fit_wall_clock_s`/`train_time_sec` fields report this total; selected-fit and selected-validation components plus a per-candidate selection log remain available.
- Contract tests assert retained `max_order` behavior, count-table size, aggregate metrics, total timing, component timing, and evaluation timing.

### GREEN and full-suite evidence

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_vomm -v
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s next_token_experiment/tests -v
git diff --check
```

Results: focused VOMM 11/11 passed; main suite 37/37 passed; next-token suite 19/19 passed; diff check passed with only the repository's CRLF conversion notices.

### Commit

Source and tests: `eaff32f fix: harden VOMM evaluation contract`.

### Remaining concerns

Superseded by fix round 2: BOS is now explicit and the no-PAD comparison support is enforced. Runner/config integration and deferred FFBS work remain owned by later tasks.

## Fix round 2/5

### Status and scope

Complete. The remaining exact-coverage and BOS/support findings were addressed in `Comparacion/vomm.py` and `tests/test_vomm.py`. Runner, config, Transformer, and documentation files were not changed.

### RED evidence

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_vomm -v
```

Result before implementation: 12 tests ran with 14 expected errors. The valid explicit-BOS constructor and all updated call sites failed with `TypeError` because `bos_token_id` was not yet accepted by `VariableOrderMarkovModel` or `select_vomm_by_validation`.

### Implemented corrections

- `bos_token_id` is now a required constructor and validation-selection argument; it is validated as a positive non-boolean integer and propagated to every candidate.
- Task 2 scoring support is explicitly musical symbol IDs `0 <= token < bos_token_id` plus BOS, with the binding invariant `vocabulary_size == bos_token_id + 1`. This excludes PAD and any other tokenizer-only symbols.
- BOS remains legal only once at prediction-context position zero and remains illegal as a training or evaluation target.
- The coverage fixture now uses five distinct musical targets `[0, 1, 2, 3, 4]`, BOS `5`, a three-token first slice, and a two-token tail. It asserts exact ordered targets, exact contexts, and exact `(target, context)` pairs: `(0, [5])`, `(1, [5, 0])`, `(2, [5, 0, 1])`, `(3, [5])`, `(4, [5, 3])`.

### Binding Task 4 integration obligation

Task 4 must construct/select this VOMM with `bos_token_id=tokenizer.bos_token_id` and `vocabulary_size=tokenizer.bos_token_id + 1`. It must keep VOMM/HMM comparison targets within the musical range below BOS and must mask or exclude Transformer PAD from shared predictive support and metrics. Task 2 intentionally does not modify the runner or Transformer.

### GREEN and full-suite evidence

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_vomm -v
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s next_token_experiment/tests -v
git diff --check
```

Results: focused VOMM 12/12 passed; main suite 38/38 passed; next-token suite 19/19 passed; diff check passed with only the repository's CRLF conversion notices.

### Commit

Source and tests: `ed6b3fe fix: make VOMM BOS support explicit`.

### Remaining concerns

Task 4 must satisfy the binding tokenizer/BOS/PAD obligation above before runner integration. No Task 2 source concern remains from this review round.
