# Task 1 Report: Protocolo predictivo justo y HDP-HMM segmentado

## Status

DONE

## Changed files

- `next_token_experiment/data/dataset.py`
- `Comparacion/classical_models.py`
- `src/models/hdp_hmm.py`
- `tests/test_comparacion.py`
- `tests/test_hdp_hmm.py`
- `next_token_experiment/tests/test_protocol.py`
- `.superpowers/sdd/2026-08-20-multidimensional-evaluation/task-1-report.md`

No canonical training, PDMX scan, dependency installation, or subagent dispatch was performed.

## Commits

- `51f631c37a7474af29376d6422fdaf6142d42e1b` — `fix: align predictive evaluation across model families`
- The report is committed in the immediate follow-up documentation commit. Its SHA is reported in the final handoff because a commit cannot contain its own final SHA.

## RED commands and observed failures

### Evaluation slices and BOS conditioning

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest next_token_experiment.tests.test_protocol tests.test_comparacion -v
```

Initial run: 5 tests ran with 2 expected contract errors: `build_evaluation_slices` could not be imported and `FiniteGlobalHMM.fit()` rejected `max_context_length`. The tests were adjusted to express absent contracts as assertion failures, without changing production code, and the same command was rerun.

Clean RED rerun: 11 tests ran with 3 failures:

- `build_evaluation_slices` was absent.
- Validation windows were `[(0, 128), (64, 129)]` instead of `[(0, 128), (128, 129)]`.
- `FiniteGlobalHMM.fit()` did not accept `max_context_length`.

### Segmented HDP-HMM fitting

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_hdp_hmm tests.test_comparacion -v
```

Observed: 7 tests ran with 2 failures. `_count_segmented_transitions` and `TruncatedHDPHMM.fit_sequences` were absent. Existing HDP-HMM and comparison tests remained green.

### Selected-result consistency

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_hdp_hmm tests.test_comparacion -v
```

Observed: 8 tests ran with 1 failure. The fit reported winning `alpha=1.0`, but `best_result` retained the losing candidate's token probability (`0.2` instead of `0.8`).

### Context validation edge case

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest next_token_experiment.tests.test_protocol.ProtocolTests.test_evaluation_slices_keep_short_tail_exactly_once -v
```

Observed: 1 test ran with 1 failure because `build_evaluation_slices(0, 1)` did not raise `ValueError`; validation occurred after the empty-sequence return.

## GREEN and full-test commands

### Focused GREEN checkpoints

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest next_token_experiment.tests.test_protocol tests.test_comparacion -v
```

Result after evaluation/BOS implementation: 11 tests, 0 failures.

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_hdp_hmm tests.test_comparacion -v
```

Result after segmented fitting: 7 tests, 0 failures. Result after selected-result fix: 8 tests, 0 failures.

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest next_token_experiment.tests.test_protocol tests.test_hdp_hmm tests.test_comparacion -v
```

Final focused acceptance result: 16 tests, 0 failures.

### Fresh final full verification

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Result: 25 tests, 0 failures, 0 errors (`OK`).

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s next_token_experiment/tests -v
```

Result: 19 tests, 0 failures, 0 errors (`OK`).

```powershell
git diff --check
```

Result: exit 0. Git printed only Windows LF-to-CRLF conversion notices.

## Design decisions

- Added `build_evaluation_slices(sequence_length, max_context_length)` with non-overlapping slices, complete positive-tail retention, and context-length validation before the empty-sequence return.
- Kept overlapping `build_window_slices` behavior for training. `WindowedSequenceDataset` uses the shared non-overlapping protocol for validation and test splits, including one-token tails regardless of `min_window_length`.
- Added `conditional_log_likelihood` to compute `log p([BOS] + targets) - log p([BOS])`. Finite HMM and HDP-HMM validation/test scoring sum this quantity over the same evaluation slices and report exact musical-token counts.
- Added `max_context_length` to finite HMM and global HDP-HMM fit/evaluate interfaces, with a backward-compatible default of 128 for existing callers.
- Implemented `TruncatedHDPHMM.fit_sequences`: state paths are initialized and sampled independently, transition counts are accumulated only within paths, and every sequence start contributes to the initial-state Dirichlet update. Concatenation is limited to emission/state summaries and the returned combined observation/result representation. `fit` delegates to a one-element `fit_sequences` call.
- Global HDP-HMM training now passes one BOS-augmented `ObservationSequence` per piece, eliminating artificial end-of-piece to next-piece transitions.
- Candidate validation scoring uses a local result. `self.best_result`, selected hyperparameters, and retained validation NLL are updated together only when validation improves.
- Deterministic tests use literal slice/count expectations and a narrow sampler double only to make hyperparameter-selection consistency independent of stochastic inference.

## Concerns

None. The user prohibition on subagents prevented an independent reviewer dispatch, so the diff and acceptance criteria were audited locally. Git's LF-to-CRLF notices are informational; `git diff --check` passed.
