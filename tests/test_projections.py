from __future__ import annotations

import jax
import numpy as np
import pytest

from jgot import GraphSpec, LogMeanOps
from jgot.linear_solvers import (
    _build_ceh_constraint_pullback,
    _build_ceh_matvec,
    _build_ceh_rhs,
    _solve_tridiagonal_javg_thomas_reference,
    solve_tridiagonal_javg,
)
from jgot.operators import continuity_residual
from jgot.projections import (
    project_ceh,
    project_javg,
    project_jeq,
    project_jpm,
    project_k,
    prox_a_star,
)


def _path_graph() -> GraphSpec:
    return GraphSpec.from_undirected_weights(3, [0, 1], [1, 2], [1.0, 1.0])


def _zero_mean_basis(dim: int) -> np.ndarray:
    raw = np.zeros((dim, dim - 1), dtype=np.float64)
    raw[:-1, :] = np.eye(dim - 1, dtype=np.float64)
    raw[-1, :] = -1.0
    basis, _ = np.linalg.qr(raw)
    return basis



def test_prox_a_star_is_identity_on_feasible_points() -> None:
    p = np.array([[-2.0]])
    q = np.array([[0.5]])
    p_pr, q_pr = prox_a_star(p, q, newton_iters=6)
    np.testing.assert_allclose(np.asarray(p_pr), p)
    np.testing.assert_allclose(np.asarray(q_pr), q)



def test_project_k_handles_inside_and_bottom_cases() -> None:
    mean = LogMeanOps()
    a = np.array([[1.0, 1.0]])
    b = np.array([[1.0, -2.0]])
    c = np.array([[0.5, -1.0]])
    a_pr, b_pr, c_pr = project_k(mean, a, b, c)
    np.testing.assert_allclose(np.asarray(a_pr)[0, 0], 1.0)
    np.testing.assert_allclose(np.asarray(b_pr)[0, 0], 1.0)
    np.testing.assert_allclose(np.asarray(c_pr)[0, 0], 0.5)
    np.testing.assert_allclose(np.asarray(c_pr)[0, 1], 0.0)
    assert float(np.asarray(a_pr)[0, 1]) >= 0.0



def test_project_jpm_enforces_node_edge_consistency() -> None:
    graph = _path_graph()
    q_node = np.ones((2, graph.num_nodes))
    rho_minus = np.zeros((2, graph.num_edges))
    rho_plus = np.zeros((2, graph.num_edges))
    q_pr, rho_minus_pr, rho_plus_pr = project_jpm(graph, q_node, rho_minus, rho_plus)
    np.testing.assert_allclose(np.asarray(rho_minus_pr), np.asarray(q_pr)[:, np.asarray(graph.src)])
    np.testing.assert_allclose(np.asarray(rho_plus_pr), np.asarray(q_pr)[:, np.asarray(graph.dst)])



def test_project_javg_enforces_time_averaging() -> None:
    rho = np.array([[1.2, 0.8], [1.0, 1.0], [0.8, 1.2]])
    rho_bar = np.array([[0.9, 1.1], [1.1, 0.9]])
    rho_a = rho[0]
    rho_b = rho[-1]
    rho_pr, rho_bar_pr = project_javg(rho, rho_bar, rho_a, rho_b)
    np.testing.assert_allclose(np.asarray(rho_pr)[0], rho_a)
    np.testing.assert_allclose(np.asarray(rho_pr)[-1], rho_b)
    np.testing.assert_allclose(
        np.asarray(rho_bar_pr),
        0.5 * (np.asarray(rho_pr[:-1]) + np.asarray(rho_pr[1:])),
    )


@pytest.mark.parametrize(("num_steps", "num_nodes"), [(2, 1), (4, 3), (7, 5)])
def test_solve_tridiagonal_javg_matches_thomas_reference(num_steps: int, num_nodes: int) -> None:
    rng = np.random.default_rng(100 * num_steps + num_nodes)
    rhs = rng.normal(size=(num_steps, num_nodes))
    rho = np.zeros((num_steps + 1, num_nodes), dtype=np.float64)
    endpoint = np.zeros((num_nodes,), dtype=np.float64)

    lam = solve_tridiagonal_javg(rhs, rho, endpoint, endpoint)
    lam_ref = _solve_tridiagonal_javg_thomas_reference(rhs)

    np.testing.assert_allclose(np.asarray(lam), np.asarray(lam_ref), atol=1e-12, rtol=1e-12)


def test_solve_tridiagonal_javg_matches_eager_under_jit() -> None:
    rhs = np.array([[0.5, -0.25], [1.0, 0.75], [-0.5, 0.25]], dtype=np.float64)
    rho = np.zeros((rhs.shape[0] + 1, rhs.shape[1]), dtype=np.float64)
    endpoint = np.zeros((rhs.shape[1],), dtype=np.float64)

    def run_once(rhs_value):
        return solve_tridiagonal_javg(rhs_value, rho, endpoint, endpoint)

    eager = run_once(rhs)
    jitted = jax.jit(run_once)(rhs)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-12, rtol=1e-12)


def test_project_javg_matches_thomas_reference_projection() -> None:
    rho = np.array([[1.2, 0.8], [1.0, 1.0], [0.8, 1.2]], dtype=np.float64)
    rho_bar = np.array([[0.9, 1.1], [1.1, 0.9]], dtype=np.float64)
    rho_a = rho[0]
    rho_b = rho[-1]

    rhs = rho_bar - 0.5 * (rho[:-1] + rho[1:])
    rhs[0] = rho_bar[0] - 0.5 * (rho_a + rho[1])
    rhs[-1] = rho_bar[-1] - 0.5 * (rho_b + rho[-2])
    lam_ref = np.asarray(_solve_tridiagonal_javg_thomas_reference(rhs))
    rho_expected = rho.copy()
    rho_expected[0] = rho_a
    rho_expected[-1] = rho_b
    rho_expected[1:-1] = rho[1:-1] + 0.5 * (lam_ref[:-1] + lam_ref[1:])
    rho_bar_expected = rho_bar - lam_ref

    rho_pr, rho_bar_pr = project_javg(rho, rho_bar, rho_a, rho_b)
    np.testing.assert_allclose(np.asarray(rho_pr), rho_expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(rho_bar_pr), rho_bar_expected, atol=1e-12, rtol=1e-12)



def test_project_jeq_is_midpoint() -> None:
    a = np.array([[1.0, 2.0]])
    b = np.array([[3.0, 4.0]])
    a_pr, b_pr = project_jeq(a, b)
    np.testing.assert_allclose(np.asarray(a_pr), [[2.0, 3.0]])
    np.testing.assert_allclose(np.asarray(b_pr), [[2.0, 3.0]])



def test_project_ceh_enforces_boundary_and_continuity() -> None:
    graph = GraphSpec.from_undirected_weights(2, [0], [1], [1.0])
    rho = np.array([[1.2, 0.8], [1.1, 0.9], [0.7, 1.3]])
    m = np.zeros((2, graph.num_edges))
    rho_a = np.array([1.5, 0.5])
    rho_b = np.array([0.5, 1.5])
    rho_pr, m_pr, _, cg_residual, cg_iters = project_ceh(
        graph,
        rho,
        m,
        rho_a,
        rho_b,
        cg_max_iters=64,
        cg_tol=1e-12,
    )
    np.testing.assert_allclose(np.asarray(rho_pr)[0], rho_a)
    np.testing.assert_allclose(np.asarray(rho_pr)[-1], rho_b)
    res = np.asarray(continuity_residual(graph, rho_pr, m_pr, rho_a, rho_b))
    assert np.max(np.abs(res)) < 1e-6
    assert float(cg_residual) < 1e-8
    assert int(cg_iters) > 0


def test_project_ceh_block_jacobi_enforces_boundary_and_continuity() -> None:
    graph = GraphSpec.from_undirected_weights(2, [0], [1], [1.0])
    rho = np.array([[1.2, 0.8], [1.1, 0.9], [0.7, 1.3]])
    m = np.zeros((2, graph.num_edges))
    rho_a = np.array([1.5, 0.5])
    rho_b = np.array([0.5, 1.5])
    rho_pr, m_pr, _, cg_residual, cg_iters = project_ceh(
        graph,
        rho,
        m,
        rho_a,
        rho_b,
        cg_max_iters=64,
        cg_tol=1e-12,
        cg_preconditioner="block_jacobi",
    )
    np.testing.assert_allclose(np.asarray(rho_pr)[0], rho_a)
    np.testing.assert_allclose(np.asarray(rho_pr)[-1], rho_b)
    res = np.asarray(continuity_residual(graph, rho_pr, m_pr, rho_a, rho_b))
    assert np.max(np.abs(res)) < 1e-6
    assert float(cg_residual) < 1e-8
    assert int(cg_iters) > 0


def test_project_ceh_block_jacobi_matches_eager_under_jit() -> None:
    graph = GraphSpec.from_undirected_weights(2, [0], [1], [1.0])
    rho = np.array([[1.2, 0.8], [1.1, 0.9], [0.7, 1.3]])
    m = np.zeros((2, graph.num_edges))
    rho_a = np.array([1.5, 0.5])
    rho_b = np.array([0.5, 1.5])

    def run_once(rho_value, m_value):
        return project_ceh(
            graph,
            rho_value,
            m_value,
            rho_a,
            rho_b,
            cg_max_iters=64,
            cg_tol=1e-12,
            cg_preconditioner="block_jacobi",
        )

    eager = run_once(rho, m)
    jitted = jax.jit(run_once)(rho, m)
    for eager_item, jit_item in zip(eager, jitted, strict=True):
        np.testing.assert_allclose(
            np.asarray(jit_item),
            np.asarray(eager_item),
            atol=1e-9,
            rtol=1e-9,
        )


def test_project_ceh_block_jacobi_matches_dense_reference_on_tiny_problem() -> None:
    graph = GraphSpec.from_undirected_weights(2, [0], [1], [1.0])
    rho = np.array([[1.2, 0.8], [1.1, 0.9], [0.7, 1.3]], dtype=np.float64)
    m = np.zeros((2, graph.num_edges), dtype=np.float64)
    rho_a = np.array([1.5, 0.5], dtype=np.float64)
    rho_b = np.array([0.5, 1.5], dtype=np.float64)

    rho_pr, m_pr, phi, _, _ = project_ceh(
        graph,
        rho,
        m,
        rho_a,
        rho_b,
        cg_max_iters=128,
        cg_tol=1e-12,
        cg_preconditioner="block_jacobi",
    )
    matvec = _build_ceh_matvec(graph, rho, m)
    rhs = np.asarray(_build_ceh_rhs(graph, rho, m, rho_a, rho_b))
    shape = rhs.shape
    size = rhs.size
    matrix = np.column_stack(
        [
            np.asarray(matvec(np.eye(size, dtype=np.float64)[k].reshape(shape))).reshape(-1)
            for k in range(size)
        ]
    )
    basis = _zero_mean_basis(size)
    phi_ref_flat = basis @ np.linalg.solve(
        basis.T @ matrix @ basis,
        basis.T @ rhs.reshape(-1),
    )
    phi_ref = phi_ref_flat.reshape(shape)

    _, pullback = _build_ceh_constraint_pullback(
        graph,
        jax.numpy.asarray(rho),
        jax.numpy.asarray(m),
    )
    drho_int_adj_ref, dm_adj_ref = pullback(jax.numpy.asarray(phi_ref))
    node_weight = np.asarray(graph.pi)[None, :]
    edge_weight = (
        0.5
        * np.asarray(graph.q)[None, :]
        * np.asarray(graph.pi)[np.asarray(graph.src)][None, :]
    )
    drho_int_ref = np.asarray(drho_int_adj_ref) / np.maximum(node_weight, 1e-30)
    dm_ref = np.asarray(dm_adj_ref) / np.maximum(edge_weight, 1e-30)
    drho_ref = np.zeros_like(rho)
    drho_ref[1:-1] = drho_int_ref
    rho_pr_ref = rho - drho_ref
    rho_pr_ref[0] = rho_a
    rho_pr_ref[-1] = rho_b
    m_pr_ref = m - dm_ref

    np.testing.assert_allclose(np.asarray(phi), phi_ref, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(np.asarray(rho_pr), rho_pr_ref, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(np.asarray(m_pr), m_pr_ref, atol=1e-8, rtol=1e-8)
