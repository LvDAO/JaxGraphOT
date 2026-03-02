from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jgot import GraphSpec, LogMeanOps, OTConfig, OTProblem, TimeDiscretization, solve_ot

DEFAULT_CONFIG = OTConfig(max_iters=800, check_every=20, cg_max_iters=128)


def ring_graph(num_nodes: int) -> GraphSpec:
    u = list(range(num_nodes))
    v = [(i + 1) % num_nodes for i in range(num_nodes)]
    w = [1.0] * num_nodes
    return GraphSpec.from_undirected_weights(num_nodes, u, v, w)



def path_graph(num_nodes: int) -> GraphSpec:
    u = list(range(num_nodes - 1))
    v = list(range(1, num_nodes))
    w = [1.0] * (num_nodes - 1)
    return GraphSpec.from_undirected_weights(num_nodes, u, v, w)



def ring_layout(num_nodes: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num_nodes, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)



def path_layout(num_nodes: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, num_nodes)
    y = np.zeros(num_nodes, dtype=np.float64)
    return np.stack([x, y], axis=1)



def dirac_density(graph: GraphSpec, node: int) -> np.ndarray:
    pi = np.asarray(graph.pi)
    rho = np.zeros(graph.num_nodes, dtype=np.float64)
    rho[node] = 1.0 / pi[node]
    return rho



def solve_problem(
    graph: GraphSpec,
    rho_a: Iterable[float],
    rho_b: Iterable[float],
    *,
    num_steps: int,
    config: OTConfig = DEFAULT_CONFIG,
):
    problem = OTProblem(
        graph=graph,
        time=TimeDiscretization(num_steps),
        rho_a=np.asarray(rho_a, dtype=np.float64),
        rho_b=np.asarray(rho_b, dtype=np.float64),
        mean_ops=LogMeanOps(),
    )
    return solve_ot(problem, config)



def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir



def save_solution(output_dir: Path, name: str, solution) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{name}.npz"
    np.savez(
        path,
        distance=np.asarray(solution.distance),
        action=np.asarray(solution.action),
        rho=np.asarray(solution.state.rho),
        m=np.asarray(solution.state.m),
        vartheta=np.asarray(solution.state.vartheta),
        rho_minus=np.asarray(solution.state.rho_minus),
        rho_plus=np.asarray(solution.state.rho_plus),
        rho_bar=np.asarray(solution.state.rho_bar),
        q_node=np.asarray(solution.state.q_node),
        converged=np.asarray(solution.converged),
        iterations_used=np.asarray(solution.iterations_used),
    )
    return path



def summarize_solution(label: str, solution) -> str:
    diagnostics = {k: float(v) for k, v in solution.diagnostics.items()}
    return (
        f"{label}: distance={float(solution.distance):.8f}, "
        f"action={float(solution.action):.8f}, converged={solution.converged}, "
        f"iterations={solution.iterations_used}, diagnostics={diagnostics}"
    )



def probability_mass(graph: GraphSpec, solution) -> np.ndarray:
    return np.asarray(solution.state.rho) * np.asarray(graph.pi)[None, :]



def unique_edge_flow(graph: GraphSpec, solution) -> np.ndarray:
    edge_idx, _, _ = undirected_edges(graph)
    flow = np.asarray(solution.state.m)[:, edge_idx]
    return flow



def undirected_edges(graph: GraphSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rev = np.asarray(graph.rev)
    edge_idx = np.where(np.arange(graph.num_edges) < rev)[0]
    src = np.asarray(graph.src)[edge_idx]
    dst = np.asarray(graph.dst)[edge_idx]
    return edge_idx, src, dst



def save_node_mass_lines(
    output_dir: Path,
    name: str,
    graph: GraphSpec,
    solution,
    *,
    title: str,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{name}_node_mass.png"
    node_mass = probability_mass(graph, solution)
    time = np.linspace(0.0, 1.0, node_mass.shape[0])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for node_idx in range(graph.num_nodes):
        ax.plot(time, node_mass[:, node_idx], linewidth=2.0, label=f"node {node_idx}")
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("probability mass")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncols=min(graph.num_nodes, 4))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path



def save_node_mass_heatmap(
    output_dir: Path,
    name: str,
    graph: GraphSpec,
    solution,
    *,
    title: str,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{name}_node_heatmap.png"
    node_mass = probability_mass(graph, solution)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(node_mass.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("node")
    ax.set_yticks(range(graph.num_nodes))
    fig.colorbar(im, ax=ax, label="probability mass")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path



def save_edge_flow_heatmap(
    output_dir: Path,
    name: str,
    graph: GraphSpec,
    solution,
    *,
    title: str,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{name}_edge_flow.png"
    flow = unique_edge_flow(graph, solution)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(flow.T, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("undirected edge")
    ax.set_yticks(range(flow.shape[1]))
    fig.colorbar(im, ax=ax, label="edge flow")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path



def save_graph_snapshot_series(
    output_dir: Path,
    name: str,
    graph: GraphSpec,
    solution,
    *,
    positions: np.ndarray,
    title: str,
    snapshot_indices: Sequence[int] | None = None,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{name}_graph_snapshots.png"
    rho = np.asarray(solution.state.rho)
    flow = unique_edge_flow(graph, solution)
    _, edge_src, edge_dst = undirected_edges(graph)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (graph.num_nodes, 2):
        raise ValueError(f"positions must have shape ({graph.num_nodes}, 2)")

    if snapshot_indices is None:
        snapshot_indices = (0, rho.shape[0] // 2, rho.shape[0] - 1)

    density_max = max(float(np.max(rho)), 1e-12)
    flow_scale = max(float(np.max(np.abs(flow))), 1e-12)

    fig, axes = plt.subplots(1, len(snapshot_indices), figsize=(4.6 * len(snapshot_indices), 4.5))
    if len(snapshot_indices) == 1:
        axes = [axes]

    node_artist = None
    for ax, time_idx in zip(axes, snapshot_indices, strict=True):
        node_values = rho[time_idx]
        edge_time = min(time_idx, flow.shape[0] - 1)
        edge_values = flow[edge_time]

        for edge_idx, (src, dst) in enumerate(zip(edge_src, edge_dst, strict=True)):
            start = positions[src]
            end = positions[dst]
            edge_value = edge_values[edge_idx]
            width = 1.5 + 5.0 * abs(edge_value) / flow_scale
            color = "#c0392b" if edge_value >= 0 else "#2471a3"
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=width,
                alpha=0.55,
                solid_capstyle="round",
                zorder=1,
            )

        node_sizes = 350.0 + 2600.0 * (node_values / density_max)
        node_artist = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=node_sizes,
            c=node_values,
            cmap="viridis",
            edgecolors="black",
            linewidths=1.0,
            zorder=2,
        )

        for node_idx, (x, y) in enumerate(positions):
            ax.text(
                x,
                y,
                f"{node_idx}\n{node_values[node_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                zorder=3,
            )

        ax.set_title(f"t = {time_idx / (rho.shape[0] - 1):.2f}")
        ax.set_aspect("equal")
        ax.axis("off")

        pad = 0.28
        ax.set_xlim(float(np.min(positions[:, 0]) - pad), float(np.max(positions[:, 0]) + pad))
        ax.set_ylim(float(np.min(positions[:, 1]) - pad), float(np.max(positions[:, 1]) + pad))

    fig.suptitle(title)
    if node_artist is not None:
        cbar = fig.colorbar(node_artist, ax=axes, fraction=0.03, pad=0.03)
        cbar.set_label("density")
    fig.subplots_adjust(left=0.03, right=0.92, bottom=0.08, top=0.84, wspace=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
