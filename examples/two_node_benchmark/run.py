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



def reference_distance(alpha: float, beta: float, samples: int = 4001) -> float:
    grid = np.linspace(alpha, beta, samples)
    ratio = np.empty_like(grid)
    mask = np.abs(grid) < 1e-12
    ratio[mask] = 1.0
    ratio[~mask] = np.arctanh(grid[~mask]) / grid[~mask]
    return (1.0 / np.sqrt(2.0)) * np.trapezoid(np.sqrt(ratio), grid)



def main() -> None:
    _bootstrap_examples_dir()

    from _common import (
        path_layout,
        save_edge_flow_heatmap,
        save_graph_snapshot_series,
        save_node_mass_lines,
        save_solution,
        solve_problem,
        summarize_solution,
    )
    from jgot import GraphSpec

    parser = argparse.ArgumentParser(description="Run the 2-node benchmark from the paper.")
    parser.add_argument("--alpha", type=float, default=-0.3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()

    graph = GraphSpec.from_undirected_weights(2, [0], [1], [1.0])
    rho_a = np.array([1.0 - args.alpha, 1.0 + args.alpha], dtype=np.float64)
    rho_b = np.array([1.0 - args.beta, 1.0 + args.beta], dtype=np.float64)
    solution = solve_problem(graph, rho_a, rho_b, num_steps=args.steps)
    reference = reference_distance(args.alpha, args.beta)
    rel_err = abs(float(solution.distance) - reference) / reference
    base_name = "two_node_benchmark"

    output = save_solution(OUTPUT_DIR, base_name, solution)
    node_plot = save_node_mass_lines(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        title="Two-node benchmark: probability mass over time",
    )
    flow_plot = save_edge_flow_heatmap(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        title="Two-node benchmark: edge flow over time",
    )
    graph_plot = save_graph_snapshot_series(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        positions=path_layout(2),
        title="Two-node benchmark: graph snapshots (node area scales with density)",
    )

    print(summarize_solution("two-node", solution))
    print(f"reference_distance={reference:.8f}, relative_error={rel_err:.8e}")
    print(f"saved_state={output}")
    print(f"saved_node_plot={node_plot}")
    print(f"saved_flow_plot={flow_plot}")
    print(f"saved_graph_plot={graph_plot}")


if __name__ == "__main__":
    main()
