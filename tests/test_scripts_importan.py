"""Los scripts de diagnóstico y de auditoría importan funciones que existen y las llaman bien.

_build_grouped_fixed_splits se reemplazó por Comparacion.splits.build_grouped_fixed_splits,
y _build_transformer_dataloaders exige model_seed y data_seed (T2).
"""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO / "scripts").rglob("*.py"))


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class ScriptImportTests(unittest.TestCase):
    def test_every_comparacion_import_in_scripts_resolves(self) -> None:
        missing = []
        for path in SCRIPTS:
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("Comparacion"):
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        if not hasattr(module, alias.name):
                            missing.append(f"{path.relative_to(REPO).as_posix()}: {node.module}.{alias.name}")
        self.assertEqual(missing, [])

    def test_transformer_dataloaders_get_both_seeds_in_every_script(self) -> None:
        bad = []
        for path in SCRIPTS:
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "id", getattr(node.func, "attr", None))
                    if name == "_build_transformer_dataloaders":
                        keywords = {keyword.arg for keyword in node.keywords}
                        if not {"model_seed", "data_seed"} <= keywords:
                            bad.append(f"{path.relative_to(REPO).as_posix()}:{node.lineno}")
        self.assertEqual(bad, [])


class ReproducirSplitTests(unittest.TestCase):
    def test_the_split_reproduction_script_runs_on_a_minimal_case(self) -> None:
        script = REPO / "scripts" / "auditoria_2026_10_26" / "comparacion" / "reproducir_split.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(20):
                rows.append(
                    {
                        "piece_id": f"0\\{index}\\p{index:02d}.mxl",
                        "source_path": f"C:\\corpus\\0\\{index}\\p{index:02d}.mxl",
                        "title": f"Obra {index}",
                        "composer": f"Autor {index}",
                        "canonical_work_id": f"autor {index} obra {index}",
                        "representation": "pitch_class",
                        "tokens": [(index * 7 + k * (index % 5 + 1)) % 12 for k in range(40)],
                        "n_events": 40,
                        "metadata": {},
                    }
                )
            cache = root / "cache.jsonl"
            cache.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            ids = [row["piece_id"] for row in rows]
            plan_dir = root / "plan"
            plan_dir.mkdir()
            (plan_dir / "execution_plan.json").write_text(
                json.dumps({"splits": {"train_pool": {"piece_ids": ids[:14]}, "validation": {"piece_ids": ids[14:17]}, "test": {"piece_ids": ids[17:]}}}),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, str(script), str(REPO), str(cache), str(plan_dir)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=300,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ids_posix=False", completed.stdout)
        self.assertIn("ids_posix=True", completed.stdout)


if __name__ == "__main__":
    unittest.main()
