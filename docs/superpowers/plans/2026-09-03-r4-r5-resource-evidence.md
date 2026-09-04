# R4/R5 Resource Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable process/GPU peak-memory measurements, a final-fit-only benchmark, and an evidence validator/package for thesis requirements R4 and R5.

**Architecture:** Reuse the existing comparison runner, model classes, artifact audit, corpus cache, and CSV writer. Add one focused resource monitor used by both the runner and benchmark; keep benchmark orchestration and requirements validation in separate modules with thin scripts.

**Tech Stack:** Python 3.12, standard library, existing `psutil`, NumPy, PyTorch, music21, unittest.

**Spec:** `E:/Tesis/documentation/receta-cierre-r4-r5.md`

## Global Constraints

- Keep `music21==9.9.1` and record that exact version.
- Do not change seeds, corpus selection, model fitting, predictions, or selection decisions.
- Missing measurements remain null with an explicit status; never encode missing as zero.
- The canonical cache must contain 3000 entries, 2933 prepared pieces, 67 exclusions, 693754 events, and SHA-256 `F42F9D7AB8550A4C366CFCF410C3CF67C85FAD46F5C4F54818403DEEC328E144`.
- Never overwrite `tesis_3000_gpu_20260823_1941` or publish corpus files/personal absolute paths.

---

### Task 1: Process and CUDA resource monitor

**Files:**
- Create: `Comparacion/resource_monitor.py`
- Create: `tests/test_resource_monitor.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `ResourceMonitor(sample_interval_s=0.05, include_children=True, use_cuda=False)` context manager and `measurement()` returning peak/status/scope fields.

- [ ] **Step 1: Write failing tests**

```python
def test_monitor_reports_positive_process_peak():
    with ResourceMonitor(sample_interval_s=0.01) as monitor:
        payload = bytearray(2_000_000)
    assert payload and monitor.measurement()["peak_process_memory_bytes"] > 0

def test_cpu_monitor_marks_cuda_not_applicable():
    with ResourceMonitor(use_cuda=False) as monitor:
        pass
    assert monitor.measurement()["peak_gpu_memory_bytes"] is None
    assert monitor.measurement()["peak_gpu_memory_status"] == "not_applicable"
```

- [ ] **Step 2: Verify the tests fail**

Run: `C:\Melodies\.venv\Scripts\python.exe -m unittest tests.test_resource_monitor -v`

Expected: import failure for `Comparacion.resource_monitor`.

- [ ] **Step 3: Implement the minimal monitor**

Use a daemon sampling thread, `psutil.Process().memory_info().rss`, recursive children, and `torch.cuda.reset_peak_memory_stats`/`max_memory_allocated` only when CUDA is requested and available. Preserve errors as status strings.

- [ ] **Step 4: Verify focused tests pass**

Run: `C:\Melodies\.venv\Scripts\python.exe -m unittest tests.test_resource_monitor -v`

Expected: all resource monitor tests pass.

### Task 2: Integrate measurements into comparison costs

**Files:**
- Modify: `Comparacion/runner.py`
- Modify: `tests/test_comparacion.py`
- Modify: `tests/test_protocol_costs.py`

**Interfaces:**
- Consumes: `ResourceMonitor`.
- Produces: raw rows and `engineering_costs.csv` fields `fit_seconds`, `evaluation_seconds`, `peak_process_memory_bytes`, `peak_process_memory_status`, `peak_gpu_memory_bytes`, `peak_gpu_memory_status`, `memory_sample_interval_s`, `memory_scope`, hardware and library versions.

- [ ] **Step 1: Extend the existing small-run assertion**

Assert the required resource/environment columns exist and CPU rows use CUDA status `not_applicable`.

- [ ] **Step 2: Verify the assertion fails**

Run: `C:\Melodies\.venv\Scripts\python.exe -m unittest tests.test_comparacion tests.test_protocol_costs -v`

- [ ] **Step 3: Wrap each family fit/evaluation in the monitor**

Store measurements on the existing raw row; do not alter model inputs, call order, or outputs. Map current protocol timing to the requested aliases without removing existing columns.

- [ ] **Step 4: Verify predictions and selection remain unchanged**

Run the focused comparison/protocol suites twice and compare deterministic metric/config fields while ignoring timing and memory fields.

### Task 3: Final-fit benchmark

**Files:**
- Create: `Comparacion/resource_benchmark.py`
- Create: `scripts/run_resource_benchmark.py`
- Create: `tests/test_resource_benchmark.py`

**Interfaces:**
- Consumes: audited source `config.json`/`results_raw.csv`, existing corpus preparation/split/model builders, and `ResourceMonitor`.
- Produces: `resource_benchmark_raw.csv`, `resource_benchmark_summary.csv`, `resource_benchmark_environment.json`, `resource_benchmark_config.json`, and `resource_benchmark_audit.json`.

- [ ] **Step 1: Test source selection and summary/audit generation**

Use a temporary source run with duplicate full-fraction rows. Select the lowest numeric model seed, one selected configuration per family, and assert median/min/max plus hashes and coverage.

- [ ] **Step 2: Verify tests fail**

Run: `C:\Melodies\.venv\Scripts\python.exe -m unittest tests.test_resource_benchmark -v`

- [ ] **Step 3: Implement benchmark core and thin CLI**

Use a single-state finite-HMM tuple, one HDP hyperparameter tuple, one VOMM order, and the recorded Transformer config so each repetition performs only final fit and evaluation. Keep seed/config fixed across repetitions.

- [ ] **Step 4: Verify focused tests and CLI help**

Run: `C:\Melodies\.venv\Scripts\python.exe -m unittest tests.test_resource_benchmark -v`

Run: `C:\Melodies\.venv\Scripts\python.exe scripts\run_resource_benchmark.py --help`

### Task 4: Requirements validator and reproducibility package

**Files:**
- Create: `Comparacion/engineering_requirements.py`
- Create: `scripts/validate_engineering_requirements.py`
- Create: `scripts/build_reproducibility_package.py`
- Create: `docs/engineering-requirements.json`
- Create: `tests/test_engineering_requirements.py`

**Interfaces:**
- Produces: `artifacts/requirements_validation.json`, `artifacts/requirements_validation.md`, a corpus-free package, and a SHA-256 manifest.

- [ ] **Step 1: Test incomplete evidence remains partial**

Build temporary evidence trees missing benchmark memory/package regeneration and assert R4/R5 are `partial`, never promoted by default.

- [ ] **Step 2: Verify tests fail**

Run: `C:\Melodies\.venv\Scripts\python.exe -m unittest tests.test_engineering_requirements -v`

- [ ] **Step 3: Implement exact file/hash/status checks**

Resolve requirement evidence relative to the repository, verify audit status and required CSV columns, and emit JSON plus Markdown from the same result object.

- [ ] **Step 4: Build a corpus-free package**

Copy only declared artifacts, reject absolute personal paths in text files, hash every included file, and include exact regeneration commands.

- [ ] **Step 5: Verify focused tests**

Run: `C:\Melodies\.venv\Scripts\python.exe -m unittest tests.test_engineering_requirements -v`

### Task 5: End-to-end verification and evidence generation

**Files:**
- Generate only under `artifacts/resource_benchmark/`, `artifacts/reproducibility/`, and `artifacts/requirements_validation.*`.

**Interfaces:**
- Consumes: canonical cache and audited source run.
- Produces: final R4/R5 evidence without editing thesis prose.

- [ ] **Step 1: Run all tests and `git diff --check`**

Run both unittest discovery commands and `git diff --check`; stop on failure.

- [ ] **Step 2: Verify canonical cache and source audit**

Check cache hash/counts and require source audit `passed` before benchmarking.

- [ ] **Step 3: Execute three benchmark repetitions alone**

Run the prescribed source run, fraction `1.0`, split seed `7`, repetitions `3`, workers `6`, and CUDA for Transformer.

- [ ] **Step 4: Build package and validate requirements**

Generate the package, manifest, JSON/Markdown validation, then report R4/R5 exactly as validated.
