# Multidimensional Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorientar Melodies hacia una comparación reproducible de estructura, predicción y costo entre HMM, HDP-HMM, VOMM y Transformer sin ejecutar experimentos pesados.

**Architecture:** El protocolo predictivo compartirá segmentación y cobertura exacta; los análisis estructural, estadístico y de costo vivirán en módulos puros separados del runner. El VOMM será un control ligero configurable. El runner ensamblará artefactos, mientras `--plan-only` permitirá auditar el trabajo antes de entrenar.

**Tech Stack:** Python 3.12, NumPy, SciPy, pandas, PyTorch, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-20-multidimensional-evaluation-design.md`

## Global Constraints

- No ejecutar entrenamientos canónicos, búsquedas de hiperparámetros, inferencia masiva ni recorridos completos de PDMX.
- Usar TDD: cada cambio de comportamiento comienza con una prueba que falla por la razón esperada.
- Mantener compatibilidad con Python 3.10 o superior y no añadir dependencias obligatorias.
- `pitch_class` es la representación principal comparable.
- `BOS` es contexto y no objetivo; cada evento de validación/prueba se puntúa exactamente una vez.
- Las afirmaciones documentales deben distinguir implementación, control opcional y experimento pendiente.

---

### Task 1: Protocolo predictivo justo y HDP-HMM segmentado

**Files:**
- Modify: `next_token_experiment/data/dataset.py`
- Modify: `Comparacion/classical_models.py`
- Modify: `src/models/hdp_hmm.py`
- Modify: `tests/test_comparacion.py`
- Modify: `next_token_experiment/tests/test_protocol.py`
- Test: `tests/test_hdp_hmm.py`

**Interfaces:**
- Produces: `build_evaluation_slices(sequence_length: int, max_context_length: int) -> list[tuple[int, int]]`.
- Produces: `conditional_log_likelihood(..., context_token: int, target_tokens: list[int]) -> float` or an equivalent private helper covered through public behavior.
- Produces: `TruncatedHDPHMM.fit_sequences(observations: list[ObservationSequence]) -> HDPHMMResult`, while `fit` remains backward compatible.
- Consumes later: all classical evaluators accept `max_context_length` and report exact token coverage.

- [ ] **Step 1: Write failing segmentation and BOS tests**

Add literal expectations:

```python
def test_evaluation_slices_keep_short_tail_exactly_once(self):
    self.assertEqual(build_evaluation_slices(150, 128), [(0, 128), (128, 150)])
    self.assertEqual(build_evaluation_slices(129, 128), [(0, 128), (128, 129)])

def test_classical_score_conditions_on_bos_without_counting_it(self):
    # Fit a tiny deterministic model, evaluate two musical tokens and assert
    # summary["n_tokens"] == 2 and score == log p(BOS,x) - log p(BOS).
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest next_token_experiment.tests.test_protocol tests.test_comparacion -v
```

Expected: failure because evaluation-tail and conditional-BOS behavior do not exist.

- [ ] **Step 3: Implement shared evaluation slices and conditional scoring**

`build_evaluation_slices` must use non-overlapping slices, reject contexts smaller than two and retain all positive tails. Validation/test datasets use it; training retains its current overlapping-window policy. HMM/HDP-HMM evaluation iterates the same slices and subtracts the likelihood of the context-only prefix.

- [ ] **Step 4: Write failing segmented-HDP tests**

Test a helper with state paths `[0, 1]` and `[2, 0]`; transition `1 -> 2` must not be counted. Test `fit_sequences` with two synthetic `ObservationSequence` objects and assert that both starts contribute to the initial-state update.

- [ ] **Step 5: Run the HDP tests and verify RED**

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_hdp_hmm tests.test_comparacion -v
```

- [ ] **Step 6: Implement segmented fitting**

Sample each sequence independently with FFBS, concatenate only for emission/state summaries, count transitions within sequences, and count one initial state per sequence. Implement `fit` as a one-element delegation to `fit_sequences`.

- [ ] **Step 7: Fix selected-result consistency**

In `GlobalHDPHMM.fit`, score a local `result`; assign `self.best_result` only inside the branch that improves validation. Add an assertion tying returned hyperparameters to the retained result in a deterministic synthetic test.

- [ ] **Step 8: Run focused and full unit suites**

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s next_token_experiment/tests -v
```

- [ ] **Step 9: Commit**

```powershell
git add next_token_experiment/data/dataset.py Comparacion/classical_models.py src/models/hdp_hmm.py tests/test_comparacion.py tests/test_hdp_hmm.py next_token_experiment/tests/test_protocol.py
git commit -m "fix: align predictive evaluation across model families"
```

### Task 2: Control puente VOMM

**Files:**
- Create: `Comparacion/vomm.py`
- Create: `tests/test_vomm.py`
- Modify: `Comparacion/__init__.py`

**Interfaces:**
- Produces: `VariableOrderMarkovModel(max_order: int, vocabulary_size: int, alpha: float = 0.5, backoff_strength: float = 1.0)`.
- Produces: `fit(sequences: list[list[int]])`, `predict_distribution(context: list[int]) -> np.ndarray`, and `evaluate(pieces, max_context_length) -> dict`.
- Produces: `select_vomm_by_validation(..., candidate_orders: tuple[int, ...])`.

- [ ] **Step 1: Write failing VOMM tests**

Use hand-derived sequences. Assert that distributions sum to one, unseen contexts back off to shorter contexts, repeated long contexts receive higher probability than the unigram alternative, and evaluation retains all tail events.

- [ ] **Step 2: Verify RED**

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_vomm -v
```

- [ ] **Step 3: Implement interpolated variable-order prediction**

For every order from zero to `max_order`, store context/next-token counts. Starting from the unigram posterior, interpolate each longer available context using

```python
weight = context_count / (context_count + backoff_strength)
local = (next_counts + alpha) / (context_count + alpha * vocabulary_size)
distribution = weight * local + (1.0 - weight) * distribution
```

Normalize defensively and never expose a zero-probability distribution.

- [ ] **Step 4: Implement validation selection and evaluation contract**

Return the same summary keys used by other models: model, selected order, validation/test NLL, perplexity, token count, parameter/count-table size, fit and evaluation wall time, plus per-piece metrics.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_vomm -v
git add Comparacion/vomm.py Comparacion/__init__.py tests/test_vomm.py
git commit -m "feat: add variable-order Markov bridge baseline"
```

### Task 3: Structural, statistical and Pareto analysis

**Files:**
- Create: `Comparacion/structural_metrics.py`
- Create: `Comparacion/statistics.py`
- Create: `Comparacion/decision.py`
- Create: `tests/test_multidimensional_analysis.py`

**Interfaces:**
- Produces: `boundary_f1(reference, predicted, tolerance) -> dict`.
- Produces: `normalized_mutual_information(reference, predicted) -> float` and `adjusted_rand_index(reference, predicted) -> float`.
- Produces: `pairwise_model_comparisons(rows, bootstrap_samples, seed) -> dict` with Holm-adjusted p-values.
- Produces: `pareto_front(rows, minimize: tuple[str, ...], maximize: tuple[str, ...]) -> list[dict]`.

- [ ] **Step 1: Write failing metric tests from literal examples**

Cover exact and tolerant boundary matches, identical/permuted cluster labels, unrelated labels, three-model pairwise comparisons, Holm monotonicity and a dominated Pareto point.

- [ ] **Step 2: Verify RED**

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_multidimensional_analysis -v
```

- [ ] **Step 3: Implement pure structural metrics**

Use one-to-one tolerant boundary matching so one predicted boundary cannot satisfy two references. Compute NMI and ARI from a contingency table without adding scikit-learn.

- [ ] **Step 4: Implement paired inference**

Aggregate repeated seeds within each `(model, piece_id)` at fraction 1.0, use the intersection of works for each model pair, bootstrap paired work-level differences, call SciPy Wilcoxon when at least two non-identical pairs exist, and adjust the family of valid p-values with Holm.

- [ ] **Step 5: Implement Pareto analysis and run tests**

Treat missing metrics as unavailable rather than best/worst. A row is dominated only when another is no worse on every available required axis and strictly better on at least one.

- [ ] **Step 6: Commit**

```powershell
git add Comparacion/structural_metrics.py Comparacion/statistics.py Comparacion/decision.py tests/test_multidimensional_analysis.py
git commit -m "feat: add multidimensional comparison metrics"
```

### Task 4: Runner, plan-only mode and project documentation

**Files:**
- Modify: `Comparacion/config.py`
- Modify: `Comparacion/runner.py`
- Modify: `Comparacion/cli.py`
- Modify: `tests/test_comparacion.py`
- Create: `docs/multidimensional-evaluation.md`
- Modify: `docs/experiments.md`
- Modify: `docs/reference/next-token-protocol.md`
- Modify: `docs/index.md`
- Modify: `README.md`

**Interfaces:**
- Adds config fields `include_vomm_control=True`, `vomm_candidate_orders=(1, 2, 4, 8)`, `bootstrap_samples=10000`, `bootstrap_seed=17`, `boundary_tolerance=1`, and `structural_annotations_path=None`.
- Adds CLI flags `--plan-only`, `--without-vomm`, and `--structural-annotations`.
- `run_learning_curve_experiment(..., plan_only: bool = False)` returns before any `fit` call when planning.

- [ ] **Step 1: Write failing plan-only and integration tests**

Build a temporary corpus of synthetic prepared pieces or patch only the corpus-preparation boundary. Assert that plan-only writes configuration, split/run counts and protocol invariants, but no model result rows. Add a lightweight one-seed/one-fraction integration test with tiny model settings; do not run a canonical Transformer.

- [ ] **Step 2: Verify RED**

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest tests.test_comparacion -v
```

- [ ] **Step 3: Integrate VOMM and artifacts**

Replace the single `_compute_wilcoxon` output with all pairwise comparisons. Write the cost table, structural status/metrics, protocol audit and Pareto summary. The audit must compare expected token indices with scored indices by model and fail fast on duplicates or omissions.

- [ ] **Step 4: Implement plan-only mode**

The plan reports pieces and groups per split, nested fractions, number of fits by model, context/reset policy, expected token coverage and estimated distinction between lightweight and neural runs. It must not instantiate or fit models.

- [ ] **Step 5: Rewrite documentation around the three-axis design**

Document the descriptive axis as primary, prediction as shared secondary axis, cost as horizontal axis, VOMM as optional diagnostic control and heavy runs as author-executed. Remove claims that smoke comparisons constitute thesis evidence.

- [ ] **Step 6: Run full verification**

```powershell
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& 'C:\Melodies\.venv\Scripts\python.exe' -m unittest discover -s next_token_experiment/tests -v
git diff --check
```

- [ ] **Step 7: Commit**

```powershell
git add Comparacion/config.py Comparacion/runner.py Comparacion/cli.py tests/test_comparacion.py docs/multidimensional-evaluation.md docs/experiments.md docs/reference/next-token-protocol.md docs/index.md README.md
git commit -m "feat: orient comparison around structure prediction and cost"
```
