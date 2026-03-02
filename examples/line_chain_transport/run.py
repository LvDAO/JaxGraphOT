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



def regularize_density(graph, rho: np.ndarray, mix: float) -> np.ndarray:
    if not 0.0 <= mix < 1.0:
        raise ValueError("mix must lie in [0, 1)")
    uniform = np.ones(graph.num_nodes, dtype=np.float64)
    return (1.0 - mix) * rho + mix * uniform



def main() -> None:
    _bootstrap_examples_dir()

    from _common import (
        dirac_density,
        path_graph,
        path_layout,
        save_edge_flow_heatmap,
        save_graph_snapshot_series,
        save_node_mass_heatmap,
        save_solution,
        solve_problem,
        summarize_solution,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run the 1D chain transport setup from the paper's simple graph experiments."
        )
    )
    parser.add_argument("--nodes", type=int, default=5, help="Number of nodes in the chain.")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument(
        "--endpoint-mix",
        type=float,
        default=1e-2,
        help="Mixing weight for uniform regularization of the endpoint Dirac masses.",
    )
    args = parser.parse_args()

    graph = path_graph(args.nodes)
    rho_a = regularize_density(graph, dirac_density(graph, 0), args.endpoint_mix)
    rho_b = regularize_density(graph, dirac_density(graph, graph.num_nodes - 1), args.endpoint_mix)
    solution = solve_problem(graph, rho_a, rho_b, num_steps=args.steps)
    base_name = f"chain_{args.nodes}_endpoints"

    output = save_solution(OUTPUT_DIR, base_name, solution)
    node_plot = save_node_mass_heatmap(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        title=f"Chain with {args.nodes} nodes: probability mass heatmap",
    )
    flow_plot = save_edge_flow_heatmap(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        title=f"Chain with {args.nodes} nodes: edge flow over time",
    )
    graph_plot = save_graph_snapshot_series(
        OUTPUT_DIR,
        base_name,
        graph,
        solution,
        positions=path_layout(args.nodes),
        title=f"Chain with {args.nodes} nodes: graph snapshots (node area scales with density)",
    )

    midpoint = np.asarray(solution.state.rho)[args.steps // 2]
    node_mass = np.asarray(graph.pi) * midpoint
    print(summarize_solution(f"chain-{args.nodes}", solution))
    print(f"endpoint_mix={args.endpoint_mix}")
    print(f"midpoint_probability_mass={node_mass}")
    print(f"saved_state={output}")
    print(f"saved_node_plot={node_plot}")
    print(f"saved_flow_plot={flow_plot}")
    print(f"saved_graph_plot={graph_plot}")


if __name__ == "__main__":
    main()
