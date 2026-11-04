"""Enmienda 11: el tope de K = 512 no es desviación, y con menos de 10 obras no hay veredicto."""

from __future__ import annotations

import pandas as pd

from src.analysis import porobra, porobra_audit


def _rows(n_works: int, shift: float = -0.5, hit: tuple = ()) -> list[dict]:
    rows = []
    for index in range(n_works):
        base = 2.0 + 0.01 * index
        rows.append(
            {
                "work_id": f"w{index}",
                "n_test": 10,
                "hdp_converged": True,
                "finite_k_hit_limit": f"w{index}" in hit,
                "hdp_test_nll": base + shift + 0.001 * index,
                "finite_test_nll": base,
                "bigram_test_nll": base,
                "unigram_test_nll": base,
            }
        )
    return rows


def _blocks(rows):
    return porobra.summary_blocks(rows, bootstrap_samples=200, seed=0, exposed_work_ids=())


def test_con_nueve_obras_no_hay_veredicto_y_con_diez_si():
    nine = porobra.compare_hdp(_rows(9), bootstrap_samples=200, seed=0)
    assert {c["verdict"] for c in nine["comparisons"]} == {"no_evaluable"}
    assert nine["min_paired_works"] == 10
    # Las cifras se siguen reportando aunque no haya veredicto.
    assert all(c["n_pairs"] == 9 and c["mean_difference"] < 0 for c in nine["comparisons"])
    ten = porobra.compare_hdp(_rows(10), bootstrap_samples=200, seed=0)
    assert {c["verdict"] for c in ten["comparisons"]} == {"mejora"}


def test_una_sensibilidad_con_pocas_obras_no_cuenta_como_cambio_de_veredicto():
    rows = _rows(40)
    for row in rows[5:]:
        row["hdp_converged"] = False
    block = _blocks(rows)["sensitivity_converged_only"]
    assert block["n_works"] == 5
    assert block["verdict_changes"] == [] and block["any_verdict_changes"] is False
    assert {pair["sensitivity"] for pair in block["verdict_not_evaluable"]} == {"no_evaluable"}


def test_la_sensibilidad_sin_techo_excluye_solo_las_obras_en_el_tope():
    blocks = _blocks(_rows(40, hit=("w3", "w7")))
    block = blocks["sensitivity_without_k_ceiling"]
    assert block["excluded_work_ids"] == ["w3", "w7"]
    assert block["n_works"] == 38
    assert "Enmienda 11" in block["rule"] and "no decide" in block["rule"]
    assert block["verdict_changes"] == []
    # La principal las conserva.
    assert blocks["comparisons"]["n_works"] == 40


def test_comparison_records_conserva_la_marca_de_techo():
    works = pd.DataFrame(
        [{"work_id": "a", "n_test": 3, "hdp_converged": True, "finite_k_hit_limit": True,
          **{f"{model}_test_nll": 2.0 for model in porobra.MODELS}}]
    )
    assert porobra_audit.comparison_records(works)[0]["finite_k_hit_limit"] is True
