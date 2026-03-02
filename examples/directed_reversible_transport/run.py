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
        unique_edge_flow,
    )
    from jgot import GraphSpec

    parser = argparse.ArgumentParser(
        description="Run a directed reversible transport example with asymmetric rates."
    )
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    graph = GraphSpec.from_directed_rates(
        3,
        src=[0, 1, 1, 2],
        dst=[1, 0, 2, 1],
        q=[2.0, 1.0, 1.0, 2.0],
    )
    mass_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    mass_b = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rho_a = mass_a / np.asarray(graph.pi)
    rho_b = mass_b / np.asarray(graph.pi)
    solution = solve_problem(graph, rho_a, rho_b, num_steps=args.steps)
    base_name = "directed_reversible_transport"

    output = save_solution(OUTPUT_DIR, base_name, solution)
    node_plot = save_node_mass_lines(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        title="Directed reversible transport: probability mass over time",
    )
    flow_plot = save_edge_flow_heatmap(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        title="Directed reversible transport: edge flow over time",
    )
    graph_plot = save_graph_snapshot_series(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        positions=path_layout(3),
        title="Directed reversible transport: graph snapshots (node area scales with density)",
    )

    midpoint_index = np.asarray(solution.state.rho).shape[0] // 2
    midpoint_mass = np.asarray(solution.state.rho)[midpoint_index] * np.asarray(graph.pi)
    first_interval_flow = unique_edge_flow(graph, solution)[0]

    print(f"inferred_pi={np.asarray(graph.pi)}")
    print(summarize_solution("directed-reversible", solution))
    print(f"midpoint_probability_mass={midpoint_mass}")
    print(f"first_interval_unique_edge_flow={first_interval_flow}")
    print(f"saved_state={output}")
    print(f"saved_node_plot={node_plot}")
    print(f"saved_flow_plot={flow_plot}")
    print(f"saved_graph_plot={graph_plot}")


if __name__ == "__main__":
    main()
