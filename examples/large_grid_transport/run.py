from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _bootstrap_examples_dir() -> None:
    examples_dir = Path(__file__).resolve().parents[1]
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))


def _block_mass(node_mass: np.ndarray, side: int, *, rows: range, cols: range) -> float:
    total = 0.0
    for row in rows:
        for col in cols:
            total += float(node_mass[row * side + col])
    return total


def main() -> None:
    _bootstrap_examples_dir()

    from _common import (
        block_density,
        estimate_state_memory_bytes,
        grid_graph,
        grid_layout,
        probability_mass,
        save_edge_flow_heatmap,
        save_graph_snapshot_series,
        save_node_mass_heatmap,
        save_solution,
        solve_problem,
        summarize_solution,
    )
    from jgot import OTConfig

    parser = argparse.ArgumentParser(
        description="Run a large square-grid transport example (default 32x32, 1024 nodes)."
    )
    parser.add_argument("--side", type=int, default=32, help="Side length of the square grid.")
    parser.add_argument("--steps", type=int, default=32, help="Number of time intervals.")
    parser.add_argument(
        "--blob-size",
        type=int,
        default=2,
        help="Side length of the source and target corner mass blocks.",
    )
    parser.add_argument("--max-iters", type=int, default=800)
    parser.add_argument("--check-every", type=int, default=20)
    parser.add_argument("--cg-max-iters", type=int, default=128)
    args = parser.parse_args()

    if args.side < 2:
        raise ValueError("side must be at least 2")
    if args.steps < 2:
        raise ValueError("steps must be at least 2")
    if args.blob_size < 1 or args.blob_size > args.side:
        raise ValueError("blob-size must lie in [1, side]")

    graph = grid_graph(args.side)
    source_rows = range(args.blob_size)
    source_cols = range(args.blob_size)
    target_rows = range(args.side - args.blob_size, args.side)
    target_cols = range(args.side - args.blob_size, args.side)
    rho_a = block_density(graph, args.side, rows=source_rows, cols=source_cols)
    rho_b = block_density(graph, args.side, rows=target_rows, cols=target_cols)
    config = OTConfig(
        max_iters=args.max_iters,
        check_every=args.check_every,
        cg_max_iters=args.cg_max_iters,
    )
    solution = solve_problem(graph, rho_a, rho_b, num_steps=args.steps, config=config)
    rho = np.asarray(solution.state.rho)

    base_name = f"large_grid_{args.side}x{args.side}"
    output = save_solution(OUTPUT_DIR, base_name, solution)
    node_plot = save_node_mass_heatmap(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        title=(
            f"Large grid transport ({args.side}x{args.side}): probability mass over time"
        ),
    )
    flow_plot = save_edge_flow_heatmap(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        title=(
            f"Large grid transport ({args.side}x{args.side}): edge flow over time"
        ),
    )
    graph_plot = save_graph_snapshot_series(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        positions=grid_layout(args.side),
        title=(
            f"Large grid transport ({args.side}x{args.side}): graph snapshots "
            "(node area scales with density)"
        ),
        snapshot_indices=(0, rho.shape[0] // 2, rho.shape[0] - 1),
    )

    node_mass = probability_mass(graph, solution)
    midpoint_mass = node_mass[node_mass.shape[0] // 2]
    source_midpoint_mass = _block_mass(
        midpoint_mass,
        args.side,
        rows=source_rows,
        cols=source_cols,
    )
    target_midpoint_mass = _block_mass(
        midpoint_mass,
        args.side,
        rows=target_rows,
        cols=target_cols,
    )
    pi = np.asarray(graph.pi)
    memory_mb = estimate_state_memory_bytes(graph, args.steps) / (1024.0 * 1024.0)

    print(
        f"grid={args.side}x{args.side}, num_nodes={graph.num_nodes}, num_edges={graph.num_edges}, "
        f"num_steps={args.steps}, blob_size={args.blob_size}"
    )
    print(f"estimated_persistent_memory_mb={memory_mb:.2f}")
    print(f"pi_min={float(np.min(pi)):.8e}, pi_max={float(np.max(pi)):.8e}")
    print(summarize_solution(f"large-grid-{args.side}x{args.side}", solution))
    print(
        "midpoint_corner_masses="
        f"{{'source': {source_midpoint_mass:.8f}, 'target': {target_midpoint_mass:.8f}}}"
    )
    print(f"saved_state={output}")
    print(f"saved_node_plot={node_plot}")
    print(f"saved_flow_plot={flow_plot}")
    print(f"saved_graph_plot={graph_plot}")


if __name__ == "__main__":
    main()
