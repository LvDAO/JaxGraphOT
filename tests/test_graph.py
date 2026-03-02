from __future__ import annotations

import numpy as np
import pytest

from jax_graph_ot import GraphSpec


def test_from_undirected_weights_builds_reversible_graph() -> None:
    graph = GraphSpec.from_undirected_weights(3, [0, 1], [1, 2], [1.0, 2.0])
    expected_pi = np.array([1.0, 3.0, 2.0]) / 6.0
    np.testing.assert_allclose(np.asarray(graph.pi), expected_pi)
    residual = np.asarray(graph.pi)[np.asarray(graph.src)] * np.asarray(graph.q)
    reverse = (
        np.asarray(graph.pi)[np.asarray(graph.dst)]
        * np.asarray(graph.q)[np.asarray(graph.rev)]
    )
    np.testing.assert_allclose(residual, reverse)


def test_from_directed_rates_infers_stationary_distribution() -> None:
    base = GraphSpec.from_undirected_weights(4, [0, 1, 2], [1, 2, 3], [1.0, 2.0, 1.5])
    inferred = GraphSpec.from_directed_rates(
        4,
        np.asarray(base.src),
        np.asarray(base.dst),
        np.asarray(base.q),
    )
    np.testing.assert_allclose(np.asarray(inferred.pi), np.asarray(base.pi), atol=1e-12)


def test_missing_reverse_edge_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing reverse edge"):
        GraphSpec.from_directed_rates(2, [0], [1], [1.0])



def test_disconnected_graph_is_rejected() -> None:
    with pytest.raises(ValueError, match="connected"):
        GraphSpec.from_undirected_weights(3, [0], [1], [1.0])



def test_nonreversible_rates_are_rejected() -> None:
    src = [0, 1, 1, 2, 2, 0]
    dst = [1, 0, 2, 1, 0, 2]
    q = [2.0, 1.0, 3.0, 1.0, 5.0, 1.0]
    with pytest.raises(ValueError, match="reversible|cycle"):
        GraphSpec.from_directed_rates(3, src, dst, q)
