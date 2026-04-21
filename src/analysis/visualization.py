from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.models.finite_hmm import CHORD_STATES, FiniteHMMResult
from src.models.hdp_hmm import HDPHMMResult
try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:  # pragma: no cover - fallback grafico cuando seaborn no esta instalado.
    sns = None
    HAS_SEABORN = False


def _ensure_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: str | Path,
    cmap: str = "viridis",
    annotate: bool = False,
) -> Path:
    path = _ensure_path(output_path)
    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(matrix, cmap=cmap, xticklabels=labels, yticklabels=labels, annot=annotate, fmt=".2f")
    else:
        plt.imshow(matrix, cmap=cmap, aspect="auto")
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_emission_heatmap(
    matrix: np.ndarray,
    state_labels: list[str],
    observation_labels: list[str],
    title: str,
    output_path: str | Path,
) -> Path:
    path = _ensure_path(output_path)
    plt.figure(figsize=(12, 8))
    if HAS_SEABORN:
        sns.heatmap(matrix, cmap="magma", xticklabels=observation_labels, yticklabels=state_labels)
    else:
        plt.imshow(matrix, cmap="magma", aspect="auto")
        plt.xticks(range(len(observation_labels)), observation_labels, rotation=90)
        plt.yticks(range(len(state_labels)), state_labels)
        plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_history(result: HDPHMMResult, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    plt.figure(figsize=(10, 4))
    plt.plot(result.diagnostics.log_likelihood_history, color="#1f77b4")
    plt.xlabel("Iteracion")
    plt.ylabel("Log-likelihood")
    plt.title("Evolucion del log-likelihood")
    plt.tight_layout()
    path_ll = output_dir / "hdp_log_likelihood.png"
    plt.savefig(path_ll, dpi=200)
    plt.close()
    paths.append(path_ll)

    plt.figure(figsize=(10, 4))
    plt.plot(result.diagnostics.active_state_history, color="#d62728")
    plt.xlabel("Iteracion")
    plt.ylabel("Estados activos")
    plt.title("Estados activos por iteracion")
    plt.tight_layout()
    path_states = output_dir / "hdp_active_states.png"
    plt.savefig(path_states, dpi=200)
    plt.close()
    paths.append(path_states)
    return paths


def plot_transition_graph(
    matrix: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    threshold: float = 0.05,
) -> Path:
    path = _ensure_path(output_path)
    graph = nx.DiGraph()
    for label in labels:
        graph.add_node(label)
    for row_index, source in enumerate(labels):
        for col_index, target in enumerate(labels):
            weight = float(matrix[row_index, col_index])
            if weight >= threshold:
                graph.add_edge(source, target, weight=weight)

    plt.figure(figsize=(10, 8))
    positions = nx.spring_layout(graph, seed=7)
    widths = [graph[u][v]["weight"] * 8 for u, v in graph.edges()]
    nx.draw_networkx_nodes(graph, positions, node_color="#ffcc66", node_size=1800)
    nx.draw_networkx_labels(graph, positions, font_size=9)
    nx.draw_networkx_edges(graph, positions, width=widths, arrows=True, arrowstyle="-|>", arrowsize=18)
    plt.title("Grafo de transiciones")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_state_timeline(
    states: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    title: str,
) -> Path:
    path = _ensure_path(output_path)
    plt.figure(figsize=(14, 3))
    plt.step(range(len(states)), states, where="post", color="#2ca02c")
    plt.yticks(sorted(set(int(state) for state in states)), [labels[int(state)] for state in sorted(set(int(state) for state in states))])
    plt.xlabel("Tiempo discreto")
    plt.ylabel("Estado latente")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def generate_finite_figures(result: FiniteHMMResult, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_labels = [CHORD_STATES[state] for state in result.active_states]
    transition_active = result.empirical_transition_matrix[np.ix_(result.active_states, result.active_states)]
    emission_active = result.emission_matrix[result.active_states]

    paths = [
        plot_heatmap(
            transition_active,
            active_labels,
            "Matriz de transicion activa del HMM finito",
            output_dir / "finite_transition_heatmap.png",
            cmap="Oranges",
            annotate=True,
        ),
        plot_emission_heatmap(
            emission_active,
            active_labels,
            result.observations.vocabulary,
            "Emisiones del HMM finito",
            output_dir / "finite_emission_heatmap.png",
        ),
        plot_transition_graph(
            transition_active,
            active_labels,
            output_dir / "finite_transition_graph.png",
        ),
        plot_state_timeline(
            result.latent_states,
            CHORD_STATES,
            output_dir / "finite_timeline.png",
            "Timeline del HMM finito",
        ),
    ]
    return paths


def generate_hdp_figures(result: HDPHMMResult, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_labels = [f"z{state}" for state in result.active_states]
    transition_active = result.posterior_transition_mean[np.ix_(result.active_states, result.active_states)]
    emission_active = result.posterior_emission_mean[result.active_states]

    paths = [
        plot_heatmap(
            transition_active,
            active_labels,
            "Matriz de transicion activa del HDP-HMM truncado",
            output_dir / "hdp_transition_heatmap.png",
            cmap="Blues",
            annotate=True,
        ),
        plot_emission_heatmap(
            emission_active,
            active_labels,
            result.observations.vocabulary,
            "Emisiones por estado activo del HDP-HMM",
            output_dir / "hdp_emission_heatmap.png",
        ),
        plot_transition_graph(
            transition_active,
            active_labels,
            output_dir / "hdp_transition_graph.png",
        ),
        plot_state_timeline(
            result.latent_states,
            [f"z{index}" for index in range(result.transition_matrix.shape[0])],
            output_dir / "hdp_timeline.png",
            "Timeline del HDP-HMM truncado",
        ),
    ]
    paths.extend(plot_history(result, output_dir))
    return paths
