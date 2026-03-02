from __future__ import annotations

import numpy as np

from jax_graph_ot import GraphSpec, LogMeanOps
from jax_graph_ot.operators import continuity_residual
from jax_graph_ot.projections import (
    project_ceh,
    project_javg,
    project_jeq,
    project_jpm,
    project_k,
    prox_a_star,
)


def _path_graph() -> GraphSpec:
    return GraphSpec.from_undirected_weights(3, [0, 1], [1, 2], [1.0, 1.0])



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
