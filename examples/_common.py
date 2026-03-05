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


def grid_graph(side: int) -> GraphSpec:
    if side < 2:
        raise ValueError("side must be at least 2")

    num_nodes = side * side
    u: list[int] = []
    v: list[int] = []
    for row in range(side):
        for col in range(side):
            node = row * side + col
            if col + 1 < side:
                u.append(node)
                v.append(node + 1)
            if row + 1 < side:
                u.append(node)
                v.append(node + side)
    w = [1.0] * len(u)
    return GraphSpec.from_undirected_weights(num_nodes, u, v, w)


def ring_layout(num_nodes: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num_nodes, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def path_layout(num_nodes: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, num_nodes)
    y = np.zeros(num_nodes, dtype=np.float64)
    return np.stack([x, y], axis=1)


def grid_layout(side: int) -> np.ndarray:
    if side < 2:
        raise ValueError("side must be at least 2")

    scale = float(side - 1)
    positions = np.empty((side * side, 2), dtype=np.float64)
    for row in range(side):
        for col in range(side):
            node = row * side + col
            positions[node, 0] = col / scale
            positions[node, 1] = 1.0 - (row / scale)
    return positions


def dirac_density(graph: GraphSpec, node: int) -> np.ndarray:
    pi = np.asarray(graph.pi)
    rho = np.zeros(graph.num_nodes, dtype=np.float64)
    rho[node] = 1.0 / pi[node]
    return rho


def block_density(
    graph: GraphSpec,
    side: int,
    rows: Sequence[int],
    cols: Sequence[int],
) -> np.ndarray:
    if side < 2:
        raise ValueError("side must be at least 2")
    if graph.num_nodes != side * side:
        raise ValueError("graph.num_nodes must equal side * side")

    row_values = list(rows)
    col_values = list(cols)
    if not row_values or not col_values:
        raise ValueError("rows and cols must each contain at least one index")

    selected: set[int] = set()
    for row in row_values:
        if row < 0 or row >= side:
            raise ValueError("row index is out of range")
        for col in col_values:
            if col < 0 or col >= side:
                raise ValueError("col index is out of range")
            selected.add(row * side + col)

    mass = np.zeros(graph.num_nodes, dtype=np.float64)
    per_node_mass = 1.0 / float(len(selected))
    for node in selected:
        mass[node] = per_node_mass
    return mass / np.asarray(graph.pi)


def estimate_state_memory_bytes(graph: GraphSpec, num_steps: int) -> int:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    return int((80 * num_steps + 40) * num_nodes + (96 * num_steps + 20) * num_edges)


def solve_problem(
    graph: GraphSpec,
    rho_a: Iterable[float],
    rho_b: Iterable[float],
    *,
    num_steps: int,
    config: OTConfig = DEFAULT_CONFIG,
):
    """Solve one OT instance for examples.

    The runtime always uses the paper-style weighted ``CE_h`` path.
    ``OTConfig.numerics_mode`` is kept for compatibility and must be ``"paper"``.
    """

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


def save_debug_trace_npz(output_dir: Path, name: str, debug_trace) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{name}_debug_trace.npz"
    num_records = int(debug_trace.num_records)
    np.savez(
        path,
        iterations=np.asarray(debug_trace.iterations),
        action=np.asarray(debug_trace.action),
        continuity_residual=np.asarray(debug_trace.continuity_residual),
        primal_delta=np.asarray(debug_trace.primal_delta),
        dual_delta=np.asarray(debug_trace.dual_delta),
        max_constraint_residual=np.asarray(debug_trace.max_constraint_residual),
        ceh_cg_residual=np.asarray(debug_trace.ceh_cg_residual),
        ceh_cg_iters=np.asarray(debug_trace.ceh_cg_iters),
        min_vartheta=np.asarray(debug_trace.min_vartheta),
        num_records=np.asarray(num_records, dtype=np.int32),
    )
    return path


def save_debug_trace_plot(output_dir: Path, name: str, debug_trace, *, title: str) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / f"{name}_debug_trace.png"
    num_records = int(debug_trace.num_records)
    iterations = np.asarray(debug_trace.iterations)[:num_records]
    action = np.asarray(debug_trace.action)[:num_records]
    continuity = np.asarray(debug_trace.continuity_residual)[:num_records]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    if num_records == 0:
        for ax in axes:
            ax.axis("off")
        fig.suptitle(title)
        axes[0].text(
            0.5,
            0.5,
            "no trace records",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    axes[0].plot(iterations, action, marker="o", linewidth=2.0, markersize=4.0)
    axes[0].set_ylabel("action")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title("Action")

    axes[1].plot(iterations, continuity, marker="o", linewidth=2.0, markersize=4.0)
    continuity_positive = np.isfinite(continuity) & (continuity > 0.0)
    if np.all(continuity_positive):
        axes[1].set_yscale("log")
    axes[1].set_ylabel("continuity residual")
    axes[1].set_xlabel("PDHG iteration")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title("Continuity Residual")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
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


def _sparse_ticks(count: int, *, max_ticks: int = 16) -> np.ndarray:
    if count <= max_ticks:
        return np.arange(count, dtype=np.int32)
    return np.linspace(0, count - 1, num=max_ticks, dtype=np.int32)


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
    ax.set_yticks(_sparse_ticks(graph.num_nodes))
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
    ax.set_yticks(_sparse_ticks(flow.shape[1]))
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
    size_scale = min(1.0, 64.0 / float(graph.num_nodes))
    label_nodes = graph.num_nodes <= 64

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
            width = (0.3 + 1.2 * size_scale) + (
                (0.8 + 4.2 * size_scale) * abs(edge_value) / flow_scale
            )
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

        node_sizes = (18.0 + 332.0 * size_scale) + (42.0 + 2558.0 * size_scale) * (
            node_values / density_max
        )
        node_artist = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=node_sizes,
            c=node_values,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.4 if label_nodes else 0.1,
            zorder=2,
        )

        if label_nodes:
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

        pad = 0.06 if graph.num_nodes > 64 else 0.28
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
