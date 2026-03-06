from __future__ import annotations

import numpy as np
import pytest

from jgot import (
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


def _directed_reversible_problem(*, num_steps: int) -> OTProblem:
    graph = GraphSpec.from_directed_rates(
        3,
        src=[0, 1, 1, 2],
        dst=[1, 0, 2, 1],
        q=[2.0, 1.0, 1.0, 2.0],
    )
    mass_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    mass_b = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return OTProblem(
        graph=graph,
        time=TimeDiscretization(num_steps),
        rho_a=mass_a / np.asarray(graph.pi),
        rho_b=mass_b / np.asarray(graph.pi),
        mean_ops=LogMeanOps(),
    )


def test_config_defaults_to_paper_mode() -> None:
    assert OTConfig().numerics_mode == "paper"


def test_config_accepts_explicit_paper_mode() -> None:
    assert OTConfig(numerics_mode="paper").numerics_mode == "paper"


def test_config_accepts_block_jacobi_preconditioner() -> None:
    assert OTConfig(cg_preconditioner="block_jacobi").cg_preconditioner == "block_jacobi"


def test_config_rejects_legacy_mode_with_migration_message() -> None:
    with pytest.raises(ValueError, match="legacy mode has been removed; use numerics_mode='paper'"):
        OTConfig(numerics_mode="legacy")


def test_solver_zero_distance_for_identical_endpoints() -> None:
    problem = _two_node_problem(0.2, 0.2, num_steps=12)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=40,
            check_every=5,
            cg_max_iters=64,
        ),
    )
    assert float(solution.distance) < 1e-6
    assert solution.converged
    assert solution.iterations_used == 1


def test_solver_zero_distance_for_identical_endpoints_in_paper_mode() -> None:
    problem = _two_node_problem(0.2, 0.2, num_steps=12)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=40,
            check_every=5,
            cg_max_iters=64,
            numerics_mode="paper",
        ),
    )
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


def test_solver_is_symmetric_on_two_node_problem_in_paper_mode() -> None:
    forward = _two_node_problem(-0.4, 0.4, num_steps=24)
    backward = _two_node_problem(0.4, -0.4, num_steps=24)
    config = OTConfig(
        max_iters=240,
        check_every=10,
        tol=1e-7,
        cg_max_iters=96,
        numerics_mode="paper",
    )
    dist_forward = float(solve_ot(forward, config).distance)
    dist_backward = float(solve_ot(backward, config).distance)
    assert abs(dist_forward - dist_backward) < 1e-6


def test_solver_matches_two_node_reference_reasonably() -> None:
    alpha = -0.3
    beta = 0.3
    problem = _two_node_problem(alpha, beta, num_steps=28)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=240,
            check_every=10,
            tol=1e-7,
            cg_max_iters=96,
        ),
    )
    reference = _reference_distance(alpha, beta)
    rel_err = abs(float(solution.distance) - reference) / reference
    assert solution.converged
    assert rel_err < 1e-3
    assert float(solution.diagnostics["continuity_residual"]) < 1e-8
    assert float(solution.diagnostics["max_constraint_residual"]) < 1e-8


def test_solver_matches_two_node_reference_reasonably_in_paper_mode() -> None:
    alpha = -0.3
    beta = 0.3
    problem = _two_node_problem(alpha, beta, num_steps=28)
    solution = solve_ot(
        problem,
        OTConfig(max_iters=320, check_every=10, tol=1e-7, cg_max_iters=128, numerics_mode="paper"),
    )
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


def test_directed_reversible_solver_preserves_constraints() -> None:
    problem = _directed_reversible_problem(num_steps=16)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=1200,
            check_every=20,
            residual_tol=1e-7,
            feasibility_tol=1e-7,
            cg_max_iters=128,
        ),
    )
    assert np.isfinite(float(solution.distance))
    assert np.isfinite(float(solution.action))
    assert float(solution.diagnostics["endpoint_residual"]) == 0.0
    assert float(solution.diagnostics["continuity_residual"]) < 1e-8
    assert float(solution.diagnostics["max_constraint_residual"]) < 1e-8


def test_solver_block_jacobi_runs_on_two_node_problem() -> None:
    alpha = -0.3
    beta = 0.3
    problem = _two_node_problem(alpha, beta, num_steps=20)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=240,
            check_every=10,
            tol=1e-7,
            cg_max_iters=64,
            cg_preconditioner="block_jacobi",
        ),
    )
    reference = _reference_distance(alpha, beta)
    rel_err = abs(float(solution.distance) - reference) / reference
    assert solution.converged
    assert rel_err < 5e-3
    assert float(solution.diagnostics["endpoint_residual"]) == 0.0
    assert float(solution.diagnostics["continuity_residual"]) < 1e-8
    assert float(solution.diagnostics["max_constraint_residual"]) < 1e-8


def test_solver_returns_debug_trace_when_enabled() -> None:
    problem = _two_node_problem(-0.2, 0.2, num_steps=12)
    solution = solve_ot(
        problem,
        OTConfig(max_iters=20, check_every=5, cg_max_iters=64, record_debug_trace=True),
    )
    trace = solution.debug_trace
    assert trace is not None
    assert trace.num_records > 0
    lengths = {
        len(np.asarray(trace.iterations)),
        len(np.asarray(trace.action)),
        len(np.asarray(trace.continuity_residual)),
        len(np.asarray(trace.primal_delta)),
        len(np.asarray(trace.dual_delta)),
        len(np.asarray(trace.max_constraint_residual)),
        len(np.asarray(trace.ceh_cg_residual)),
        len(np.asarray(trace.ceh_cg_iters)),
        len(np.asarray(trace.min_vartheta)),
    }
    assert len(lengths) == 1
    valid_iterations = np.asarray(trace.iterations)[: trace.num_records]
    assert np.all(np.diff(valid_iterations) > 0)


def test_solver_omits_debug_trace_when_disabled() -> None:
    problem = _two_node_problem(-0.2, 0.2, num_steps=12)
    solution = solve_ot(
        problem,
        OTConfig(max_iters=20, check_every=5, cg_max_iters=64, record_debug_trace=False),
    )
    assert solution.debug_trace is None


def test_debug_trace_records_action_and_continuity() -> None:
    problem = _two_node_problem(-0.2, 0.2, num_steps=12)
    solution = solve_ot(
        problem,
        OTConfig(max_iters=20, check_every=5, cg_max_iters=64, record_debug_trace=True),
    )
    trace = solution.debug_trace
    assert trace is not None
    action = np.asarray(trace.action)[: trace.num_records]
    continuity = np.asarray(trace.continuity_residual)[: trace.num_records]
    assert action.shape == continuity.shape
    assert np.all(np.isfinite(continuity))


def test_debug_trace_path_still_runs_under_jit() -> None:
    problem = _two_node_problem(-0.2, 0.2, num_steps=12)
    solution = solve_ot(
        problem,
        OTConfig(max_iters=20, check_every=5, cg_max_iters=64, record_debug_trace=True),
    )
    assert solution.debug_trace is not None


def test_solver_returns_debug_trace_in_paper_mode() -> None:
    problem = _two_node_problem(-0.2, 0.2, num_steps=12)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=20,
            check_every=5,
            cg_max_iters=64,
            record_debug_trace=True,
            numerics_mode="paper",
        ),
    )
    trace = solution.debug_trace
    assert trace is not None
    assert trace.num_records > 0
    assert len(np.asarray(trace.iterations)) == len(np.asarray(trace.action))
