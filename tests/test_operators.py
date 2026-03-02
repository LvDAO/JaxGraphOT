from __future__ import annotations

import numpy as np

from jax_graph_ot import GraphSpec
from jax_graph_ot.operators import avg_time, continuity_residual, div, grad, laplace


def _two_node_graph() -> GraphSpec:
    return GraphSpec.from_undirected_weights(2, [0], [1], [1.0])



def test_grad_and_div_match_two_node_laplacian() -> None:
    graph = _two_node_graph()
    phi = np.array([[1.0, 3.0]])
    g = np.asarray(grad(graph, phi))
    np.testing.assert_allclose(g, [[-2.0, 2.0]])
    np.testing.assert_allclose(np.asarray(div(graph, g)), [[2.0, -2.0]])
    np.testing.assert_allclose(np.asarray(laplace(graph, phi)), [[2.0, -2.0]])



def test_avg_time_and_zero_continuity_residual() -> None:
    graph = _two_node_graph()
    rho = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(np.asarray(avg_time(rho)), np.ones((2, 2)))
    m = np.zeros((2, graph.num_edges))
    res = np.asarray(continuity_residual(graph, rho, m, rho[0], rho[-1]))
    np.testing.assert_allclose(res, 0.0)
