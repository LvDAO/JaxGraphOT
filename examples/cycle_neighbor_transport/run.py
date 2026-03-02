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



def run_cycle(num_nodes: int, steps: int) -> None:
    from _common import (
        dirac_density,
        ring_graph,
        ring_layout,
        save_edge_flow_heatmap,
        save_graph_snapshot_series,
        save_node_mass_lines,
        save_solution,
        solve_problem,
        summarize_solution,
    )

    graph = ring_graph(num_nodes)
    rho_a = dirac_density(graph, 0)
    rho_b = dirac_density(graph, 1)
    solution = solve_problem(graph, rho_a, rho_b, num_steps=steps)
    base_name = f"cycle_{num_nodes}_neighbor"

    output = save_solution(OUTPUT_DIR, base_name, solution)
    node_plot = save_node_mass_lines(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        title=f"Cycle with {num_nodes} nodes: probability mass over time",
    )
    flow_plot = save_edge_flow_heatmap(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        title=f"Cycle with {num_nodes} nodes: edge flow over time",
    )
    graph_plot = save_graph_snapshot_series(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        positions=ring_layout(num_nodes),
        title=f"Cycle with {num_nodes} nodes: graph snapshots (node area scales with density)",
    )

    midpoint = np.asarray(solution.state.rho)[steps // 2]
    node_mass = np.asarray(graph.pi) * midpoint
    print(summarize_solution(f"cycle-{num_nodes}", solution))
    print(f"midpoint_probability_mass={node_mass}")
    if num_nodes == 3:
        print(f"third_node_midpoint_mass={node_mass[2]}")
    if num_nodes == 4:
        print(f"long_path_midpoint_mass={node_mass[2] + node_mass[3]}")
    print(f"saved_state={output}")
    print(f"saved_node_plot={node_plot}")
    print(f"saved_flow_plot={flow_plot}")
    print(f"saved_graph_plot={graph_plot}")



def main() -> None:
    _bootstrap_examples_dir()

    parser = argparse.ArgumentParser(
        description="Run the 3-cycle and 4-cycle neighbor transport setups."
    )
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="*",
        default=[3, 4],
        help="Cycle sizes to run (defaults to 3 and 4).",
    )
    args = parser.parse_args()

    for num_nodes in args.nodes:
        run_cycle(num_nodes, args.steps)


if __name__ == "__main__":
    main()
