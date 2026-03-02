from __future__ import annotations

import numpy as np

from jax_graph_ot import (
    GraphSpec,
    LogMeanOps,
    OTConfig,
    OTProblem,
    TimeDiscretization,
    solve_ot,
)


def _two_node_problem(alpha: float, beta: float, *, num_steps: int = 28) -> OTProblem:
    graph = GraphSpec.from_undirected_weights(2, [0], [1], [1.0])
    rho_a = np.array([1.0 - alpha, 1.0 + alpha])
    rho_b = np.array([1.0 - beta, 1.0 + beta])
    return OTProblem(
        graph=graph,
        time=TimeDiscretization(num_steps),
        rho_a=rho_a,
        rho_b=rho_b,
        mean_ops=LogMeanOps(),
    )


def _reference_distance(alpha: float, beta: float, n: int = 4001) -> float:
    grid = np.linspace(alpha, beta, n)
    abs_grid = np.abs(grid)
    ratio = np.empty_like(grid)
    mask = abs_grid < 1e-10
    ratio[mask] = 1.0
    ratio[~mask] = np.arctanh(grid[~mask]) / grid[~mask]
    integrand = np.sqrt(ratio)
    return (1.0 / np.sqrt(2.0)) * np.trapezoid(integrand, grid)


def _cycle_problem(num_nodes: int, *, num_steps: int) -> OTProblem:
    u = list(range(num_nodes))
    v = [(i + 1) % num_nodes for i in range(num_nodes)]
    graph = GraphSpec.from_undirected_weights(num_nodes, u, v, [1.0] * num_nodes)
    rho_a = np.zeros(num_nodes)
    rho_b = np.zeros(num_nodes)
    rho_a[0] = num_nodes
    rho_b[1] = num_nodes
    return OTProblem(
        graph=graph,
        time=TimeDiscretization(num_steps),
        rho_a=rho_a,
        rho_b=rho_b,
        mean_ops=LogMeanOps(),
    )


def test_solver_zero_distance_for_identical_endpoints() -> None:
    problem = _two_node_problem(0.2, 0.2, num_steps=12)
    solution = solve_ot(problem, OTConfig(max_iters=40, check_every=5, cg_max_iters=64))
    assert float(solution.distance) < 1e-6
    assert solution.converged
    assert solution.iterations_used == 1


def test_solver_is_symmetric_on_two_node_problem() -> None:
    forward = _two_node_problem(-0.4, 0.4, num_steps=24)
    backward = _two_node_problem(0.4, -0.4, num_steps=24)
    config = OTConfig(max_iters=240, check_every=10, tol=1e-7, cg_max_iters=96)
    dist_forward = float(solve_ot(forward, config).distance)
    dist_backward = float(solve_ot(backward, config).distance)
    assert abs(dist_forward - dist_backward) < 1e-6


def test_solver_matches_two_node_reference_reasonably() -> None:
    alpha = -0.3
    beta = 0.3
    problem = _two_node_problem(alpha, beta, num_steps=28)
    solution = solve_ot(problem, OTConfig(max_iters=240, check_every=10, tol=1e-7, cg_max_iters=96))
    reference = _reference_distance(alpha, beta)
    rel_err = abs(float(solution.distance) - reference) / reference
    assert solution.converged
    assert rel_err < 1e-3
    assert float(solution.diagnostics["continuity_residual"]) < 1e-8
    assert float(solution.diagnostics["max_constraint_residual"]) < 1e-8


def test_three_cycle_routes_positive_mass_through_third_node() -> None:
    problem = _cycle_problem(3, num_steps=16)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=600,
            check_every=20,
            residual_tol=1e-7,
            feasibility_tol=1e-7,
            cg_max_iters=96,
        ),
    )
    midpoint = np.asarray(solution.state.rho)[problem.time.num_steps // 2]
    assert midpoint[2] > 1e-4
    assert float(solution.diagnostics["continuity_residual"]) < 1e-8


def test_four_cycle_keeps_long_path_mass_small() -> None:
    problem = _cycle_problem(4, num_steps=16)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=1200,
            check_every=20,
            residual_tol=1e-7,
            feasibility_tol=1e-7,
            cg_max_iters=96,
        ),
    )
    midpoint = np.asarray(solution.state.rho)[problem.time.num_steps // 2]
    assert midpoint[2] + midpoint[3] < 1e-2
    assert float(solution.diagnostics["continuity_residual"]) < 1e-8
