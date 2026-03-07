from __future__ import annotations

import inspect

import jax
import numpy as np
import pytest

from jgot import GraphSpec
from jgot.graph import (
    _build_bfs_tree_edge_order,
    _infer_pi_from_reversible_rates_kernel,
    _infer_pi_from_reversible_rates_kernel_jit,
)


def _infer_pi_reference(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    rev: np.ndarray,
    q: np.ndarray,
    tree_edge_order: np.ndarray,
) -> tuple[np.ndarray, float]:
    log_pi = np.zeros(num_nodes, dtype=np.float64)
    log_q = np.log(q)
    for edge_idx in tree_edge_order:
        x = int(src[edge_idx])
        y = int(dst[edge_idx])
        log_pi[y] = log_pi[x] + log_q[edge_idx] - log_q[rev[edge_idx]]
    shifted = log_pi - np.max(log_pi)
    pi = np.exp(shifted)
    pi /= np.sum(pi)
    edge_residual = (log_pi[dst] - log_pi[src]) - (log_q - log_q[rev])
    return pi, float(np.max(np.abs(edge_residual)))


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


def test_reversible_pi_kernel_matches_jit_on_directed_example() -> None:
    src = np.array([0, 1, 1, 2], dtype=np.int32)
    dst = np.array([1, 0, 2, 1], dtype=np.int32)
    q = np.array([2.0, 1.0, 1.0, 2.0], dtype=np.float64)
    rev = np.array([1, 0, 3, 2], dtype=np.int32)
    tree_edge_order = _build_bfs_tree_edge_order(3, src, dst)

    eager_pi, eager_residual = _infer_pi_from_reversible_rates_kernel(
        src,
        dst,
        rev,
        q,
        tree_edge_order,
    )
    jitted_pi, jitted_residual = _infer_pi_from_reversible_rates_kernel_jit(
        src,
        dst,
        rev,
        q,
        tree_edge_order,
    )

    np.testing.assert_allclose(np.asarray(jitted_pi), np.asarray(eager_pi), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(float(jitted_residual), float(eager_residual), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(np.asarray(jitted_pi), np.array([0.25, 0.5, 0.25]), atol=1e-12)
    assert float(jitted_residual) < 1e-12


def test_reversible_pi_kernel_matches_jit_on_larger_graph() -> None:
    base = GraphSpec.from_undirected_weights(
        6,
        [0, 0, 1, 2, 3, 4],
        [1, 2, 3, 3, 4, 5],
        [1.0, 3.0, 0.5, 1.25, 2.0, 0.75],
    )
    src = np.asarray(base.src)
    dst = np.asarray(base.dst)
    rev = np.asarray(base.rev)
    q = np.asarray(base.q)
    tree_edge_order = _build_bfs_tree_edge_order(base.num_nodes, src, dst)

    eager_pi, eager_residual = _infer_pi_from_reversible_rates_kernel(
        src,
        dst,
        rev,
        q,
        tree_edge_order,
    )
    jitted_pi, jitted_residual = _infer_pi_from_reversible_rates_kernel_jit(
        src,
        dst,
        rev,
        q,
        tree_edge_order,
    )

    np.testing.assert_allclose(np.asarray(eager_pi), np.asarray(base.pi), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(jitted_pi), np.asarray(eager_pi), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(float(jitted_residual), float(eager_residual), atol=1e-12, rtol=0.0)
    assert float(jitted_residual) < 1e-12


def test_reversible_pi_kernel_remains_jittable_when_wrapped() -> None:
    src = np.array([0, 1, 1, 2], dtype=np.int32)
    dst = np.array([1, 0, 2, 1], dtype=np.int32)
    q = np.array([2.0, 1.0, 1.0, 2.0], dtype=np.float64)
    rev = np.array([1, 0, 3, 2], dtype=np.int32)
    tree_edge_order = _build_bfs_tree_edge_order(3, src, dst)

    wrapped = jax.jit(_infer_pi_from_reversible_rates_kernel)
    eager_pi, eager_residual = _infer_pi_from_reversible_rates_kernel(
        src,
        dst,
        rev,
        q,
        tree_edge_order,
    )
    wrapped_pi, wrapped_residual = wrapped(src, dst, rev, q, tree_edge_order)

    np.testing.assert_allclose(np.asarray(wrapped_pi), np.asarray(eager_pi), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(float(wrapped_residual), float(eager_residual), atol=1e-12, rtol=0.0)


def test_undirected_constructor_docstring_mentions_unit_exit_rate() -> None:
    doc = inspect.getdoc(GraphSpec.from_undirected_weights)
    assert doc is not None
    assert "unit-exit-rate random walk" in doc


def test_from_undirected_weights_normalizes_each_row_sum_to_one() -> None:
    graph = GraphSpec.from_undirected_weights(4, [0, 1, 2], [1, 2, 3], [1.0, 2.0, 1.5])
    row_sum = np.zeros(graph.num_nodes, dtype=np.float64)
    np.add.at(row_sum, np.asarray(graph.src), np.asarray(graph.q))
    np.testing.assert_allclose(row_sum, np.ones(graph.num_nodes), atol=1e-12)


def test_directed_reversible_example_infers_expected_stationary_distribution() -> None:
    graph = GraphSpec.from_directed_rates(
        3,
        src=[0, 1, 1, 2],
        dst=[1, 0, 2, 1],
        q=[2.0, 1.0, 1.0, 2.0],
    )
    np.testing.assert_allclose(np.asarray(graph.pi), np.array([0.25, 0.5, 0.25]), atol=1e-12)


def test_from_directed_rates_keeps_supplied_stationary_distribution() -> None:
    base = GraphSpec.from_undirected_weights(4, [0, 1, 2], [1, 2, 3], [1.0, 2.0, 1.5])
    graph = GraphSpec.from_directed_rates(
        4,
        np.asarray(base.src),
        np.asarray(base.dst),
        np.asarray(base.q),
        pi=np.asarray(base.pi),
    )
    np.testing.assert_allclose(np.asarray(graph.pi), np.asarray(base.pi), atol=1e-12)


def test_from_directed_rates_rejects_disconnected_graph() -> None:
    src = [0, 1, 2, 3]
    dst = [1, 0, 3, 2]
    q = [1.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="connected"):
        GraphSpec.from_directed_rates(4, src, dst, q)


def test_reversible_pi_kernel_handles_strongly_varying_rates() -> None:
    pi_true = np.array([1e-6, 1e-2, 1.0, 1e2], dtype=np.float64)
    pi_true /= np.sum(pi_true)
    conductance = np.array([1.0, 0.5, 2.0], dtype=np.float64)
    src = np.array([0, 1, 1, 2, 2, 3], dtype=np.int32)
    dst = np.array([1, 0, 2, 1, 3, 2], dtype=np.int32)
    rev = np.array([1, 0, 3, 2, 5, 4], dtype=np.int32)
    q = np.array(
        [
            conductance[0] / pi_true[0],
            conductance[0] / pi_true[1],
            conductance[1] / pi_true[1],
            conductance[1] / pi_true[2],
            conductance[2] / pi_true[2],
            conductance[2] / pi_true[3],
        ],
        dtype=np.float64,
    )
    tree_edge_order = _build_bfs_tree_edge_order(4, src, dst)
    ref_pi, ref_residual = _infer_pi_reference(4, src, dst, rev, q, tree_edge_order)
    eager_pi, eager_residual = _infer_pi_from_reversible_rates_kernel(
        src,
        dst,
        rev,
        q,
        tree_edge_order,
    )
    jitted_pi, jitted_residual = _infer_pi_from_reversible_rates_kernel_jit(
        src,
        dst,
        rev,
        q,
        tree_edge_order,
    )

    assert np.all(np.isfinite(np.asarray(eager_pi)))
    assert np.all(np.asarray(eager_pi) > 0.0)
    assert np.all(np.isfinite(np.asarray(jitted_pi)))
    assert np.all(np.asarray(jitted_pi) > 0.0)
    np.testing.assert_allclose(np.asarray(eager_pi), ref_pi, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(jitted_pi), ref_pi, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(jitted_pi), pi_true, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(float(eager_residual), ref_residual, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(float(jitted_residual), ref_residual, atol=1e-12, rtol=0.0)
    assert ref_residual < 1e-9


def test_missing_reverse_edge_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing reverse edge"):
        GraphSpec.from_directed_rates(2, [0], [1], [1.0])


@pytest.mark.parametrize("rate", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("check_reversible", [True, False])
def test_directed_rates_reject_nonfinite_edge_rates(rate: float, check_reversible: bool) -> None:
    with pytest.raises(ValueError, match="rates must be finite"):
        GraphSpec.from_directed_rates(
            2,
            [0, 1],
            [1, 0],
            [rate, 1.0],
            pi=[0.5, 0.5],
            check_reversible=check_reversible,
        )


def test_disconnected_graph_is_rejected() -> None:
    with pytest.raises(ValueError, match="connected"):
        GraphSpec.from_undirected_weights(3, [0], [1], [1.0])


def test_nonreversible_rates_are_rejected() -> None:
    src = [0, 1, 1, 2, 2, 0]
    dst = [1, 0, 2, 1, 0, 2]
    q = [2.0, 1.0, 3.0, 1.0, 5.0, 1.0]
    with pytest.raises(ValueError, match="reversible|cycle"):
        GraphSpec.from_directed_rates(3, src, dst, q)
