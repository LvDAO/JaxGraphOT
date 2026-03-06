"""Graph validation and :class:`jgot.GraphSpec` construction.

Structural preprocessing still lives on the Python/NumPy side. Reversible
stationary-distribution inference uses a host-built BFS tree together with an
internal JAX kernel for the numeric log-ratio propagation.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from .types import GraphSpec


def _as_int_array(values: Iterable[int]) -> np.ndarray:
    """Coerce a one-dimensional integer-like input into a NumPy array."""

    arr = np.asarray(values, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("expected a one-dimensional integer array")
    return arr


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    """Coerce a one-dimensional float-like input into a NumPy array."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("expected a one-dimensional float array")
    return arr


def _build_reverse_edge_map(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Build the reverse-edge lookup and enforce paired directed edges.

    Raises:
        ValueError: If self-loops are present, a directed edge is duplicated, a
            reverse edge is missing, or the final reverse map is not involutive.
    """

    edge_index: dict[tuple[int, int], int] = {}
    for idx, (x, y) in enumerate(zip(src.tolist(), dst.tolist(), strict=True)):
        if x == y:
            raise ValueError("self-loops are not supported")
        key = (x, y)
        if key in edge_index:
            raise ValueError(f"duplicate directed edge {key}")
        edge_index[key] = idx

    rev = np.empty_like(src)
    for idx, (x, y) in enumerate(zip(src.tolist(), dst.tolist(), strict=True)):
        key = (y, x)
        if key not in edge_index:
            raise ValueError(f"missing reverse edge for {(x, y)}")
        rev[idx] = edge_index[key]

    if not np.all(rev[rev] == np.arange(src.size, dtype=rev.dtype)):
        raise ValueError("reverse-edge mapping is not involutive")
    return rev


def _check_connected(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> None:
    """Ensure the undirected graph support is connected.

    Raises:
        ValueError: If the graph has no edges or contains more than one
            connected component.
    """

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for x, y in zip(src.tolist(), dst.tolist(), strict=True):
        adjacency[x].append(y)
        adjacency[y].append(x)

    start = next((i for i, nbrs in enumerate(adjacency) if nbrs), None)
    if start is None:
        raise ValueError("graph must have at least one edge")

    seen = np.zeros(num_nodes, dtype=bool)
    queue: deque[int] = deque([start])
    seen[start] = True
    while queue:
        node = queue.popleft()
        for nbr in adjacency[node]:
            if not seen[nbr]:
                seen[nbr] = True
                queue.append(nbr)

    if not np.all(seen):
        raise ValueError("graph must be connected")


def _build_bfs_tree_edge_order(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Build a root-0 BFS tree as a fixed directed-edge order.

    The returned array has length ``num_nodes - 1`` and lists the directed edge
    used to discover each non-root node. Connectivity is re-checked from root 0
    so the numeric JAX kernel can assume a complete tree order.
    """

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge_idx, x in enumerate(src.tolist()):
        adjacency[x].append(edge_idx)

    seen = np.zeros(num_nodes, dtype=bool)
    seen[0] = True
    queue: deque[int] = deque([0])
    tree_edge_order = np.empty(num_nodes - 1, dtype=np.int32)
    num_tree_edges = 0

    while queue:
        node = queue.popleft()
        for edge_idx in adjacency[node]:
            nbr = int(dst[edge_idx])
            if not seen[nbr]:
                seen[nbr] = True
                tree_edge_order[num_tree_edges] = edge_idx
                num_tree_edges += 1
                queue.append(nbr)

    if num_tree_edges != num_nodes - 1 or not np.all(seen):
        raise ValueError("graph must be connected")
    return tree_edge_order


def _normalize_pi(pi: np.ndarray) -> np.ndarray:
    """Normalize a positive stationary distribution candidate to unit mass.

    Raises:
        ValueError: If ``pi`` is not one-dimensional or contains non-positive
            or non-finite total mass.
    """

    if pi.ndim != 1:
        raise ValueError("pi must be one-dimensional")
    if np.any(pi <= 0):
        raise ValueError("pi must be strictly positive")
    total = float(np.sum(pi))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("pi must have positive finite mass")
    return pi / total


def _out_rate(num_nodes: int, src: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the outgoing rate sum at each node."""

    out = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(out, src, q)
    return out


def _validate_stationarity(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    q: np.ndarray,
    pi: np.ndarray,
    out_rate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Validate that ``pi`` is stationary for the supplied directed rates.

    Raises:
        ValueError: If the nodewise stationarity residual exceeds the requested
            tolerance.
    """

    inflow = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(inflow, dst, pi[src] * q)
    residual = inflow - pi * out_rate
    scale = max(1.0, float(np.max(np.abs(pi[src] * q), initial=0.0)))
    tol = atol + rtol * scale
    if float(np.max(np.abs(residual), initial=0.0)) > tol:
        raise ValueError("pi is not stationary for the supplied rates")


def _validate_reversibility(
    src: np.ndarray,
    dst: np.ndarray,
    rev: np.ndarray,
    q: np.ndarray,
    pi: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Validate the detailed-balance condition edge by edge.

    Raises:
        ValueError: If the detailed-balance residual exceeds the requested
            tolerance.
    """

    edge_residual = pi[src] * q - pi[dst] * q[rev]
    scale = max(1.0, float(np.max(np.abs(pi[src] * q), initial=0.0)))
    tol = atol + rtol * scale
    if float(np.max(np.abs(edge_residual), initial=0.0)) > tol:
        raise ValueError("graph is not reversible under the supplied stationary distribution")


def _infer_pi_from_reversible_rates_kernel(
    src: jax.Array,
    dst: jax.Array,
    rev: jax.Array,
    q: jax.Array,
    tree_edge_order: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Infer ``pi`` numerically from a fixed BFS tree order."""

    src = jnp.asarray(src, dtype=jnp.int32)
    dst = jnp.asarray(dst, dtype=jnp.int32)
    rev = jnp.asarray(rev, dtype=jnp.int32)
    q = jnp.asarray(q, dtype=jnp.float64)
    tree_edge_order = jnp.asarray(tree_edge_order, dtype=jnp.int32)

    num_nodes = tree_edge_order.shape[0] + 1
    log_q = jnp.log(q)
    log_pi0 = jnp.zeros((num_nodes,), dtype=log_q.dtype)

    def propagate(log_pi: jax.Array, edge_idx: jax.Array) -> tuple[jax.Array, None]:
        x = src[edge_idx]
        y = dst[edge_idx]
        implied = log_pi[x] + log_q[edge_idx] - log_q[rev[edge_idx]]
        return log_pi.at[y].set(implied), None

    log_pi, _ = jax.lax.scan(propagate, log_pi0, tree_edge_order)
    shifted = log_pi - jnp.max(log_pi)
    weights = jnp.exp(shifted)
    pi = weights / jnp.sum(weights)
    edge_residual = (log_pi[dst] - log_pi[src]) - (log_q - log_q[rev])
    max_abs_edge_residual = jnp.max(jnp.abs(edge_residual))
    return pi, max_abs_edge_residual


_infer_pi_from_reversible_rates_kernel_jit = jax.jit(_infer_pi_from_reversible_rates_kernel)


def _infer_pi_from_reversible_rates(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    rev: np.ndarray,
    q: np.ndarray,
    *,
    tol_ratio: float,
) -> np.ndarray:
    """Infer ``pi`` from reversible directed rates using a JAX numeric kernel.

    Raises:
        ValueError: If cycle consistency fails or the graph is disconnected.
    """

    tree_edge_order = _build_bfs_tree_edge_order(num_nodes, src, dst)
    pi, max_abs_edge_residual = _infer_pi_from_reversible_rates_kernel_jit(
        src=jnp.asarray(src, dtype=jnp.int32),
        dst=jnp.asarray(dst, dtype=jnp.int32),
        rev=jnp.asarray(rev, dtype=jnp.int32),
        q=jnp.asarray(q, dtype=jnp.float64),
        tree_edge_order=jnp.asarray(tree_edge_order, dtype=jnp.int32),
    )
    if float(max_abs_edge_residual) > tol_ratio:
        raise ValueError("cycle inconsistency detected while inferring stationary distribution")
    return _normalize_pi(np.asarray(pi))


def _finalize_graph(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    rev: np.ndarray,
    q: np.ndarray,
    pi: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> GraphSpec:
    """Run final graph invariants and convert validated arrays into ``GraphSpec``."""

    out_rate = _out_rate(num_nodes, src, q)
    _validate_reversibility(src, dst, rev, q, pi, atol=atol, rtol=rtol)
    _validate_stationarity(num_nodes, src, dst, q, pi, out_rate, atol=atol, rtol=rtol)
    return GraphSpec(
        num_nodes=num_nodes,
        num_edges=int(src.size),
        src=jnp.asarray(src, dtype=jnp.int32),
        dst=jnp.asarray(dst, dtype=jnp.int32),
        rev=jnp.asarray(rev, dtype=jnp.int32),
        q=jnp.asarray(q, dtype=jnp.float64),
        pi=jnp.asarray(pi, dtype=jnp.float64),
        out_rate=jnp.asarray(out_rate, dtype=jnp.float64),
    )


def build_graph_from_undirected_weights(
    num_nodes: int,
    edge_u: Iterable[int],
    edge_v: Iterable[int],
    weight: Iterable[float],
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> GraphSpec:
    """Construct a validated graph from undirected conductances.

    This helper computes weighted degrees, derives ``pi`` from those degrees,
    and normalizes the off-diagonal edge weights by row sum. The returned
    ``GraphSpec`` therefore stores the off-diagonal part of a unit-exit-rate
    reversible jump process.

    Args:
        num_nodes: Total number of nodes in the graph.
        edge_u: First endpoint of each undirected edge.
        edge_v: Second endpoint of each undirected edge.
        weight: Positive conductance assigned to each undirected edge.
        atol: Absolute tolerance used by the invariant checks.
        rtol: Relative tolerance used by the invariant checks.

    Returns:
        A validated :class:`GraphSpec` with paired directed edges and a
        stationary distribution derived from weighted degree normalization.

    Raises:
        ValueError: If the inputs are malformed, contain invalid node ids,
            contain non-positive weights, contain self-loops, or define a
            disconnected graph.
    """

    u = _as_int_array(edge_u)
    v = _as_int_array(edge_v)
    w = _as_float_array(weight)
    if not (u.size == v.size == w.size):
        raise ValueError("edge_u, edge_v, and weight must have the same length")
    if num_nodes <= 1:
        raise ValueError("num_nodes must be at least 2")
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive")
    if np.any((u < 0) | (u >= num_nodes) | (v < 0) | (v >= num_nodes)):
        raise ValueError("edge endpoints are out of range")
    if np.any(u == v):
        raise ValueError("self-loops are not supported")

    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    _check_connected(num_nodes, src, dst)
    degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(degree, u, w)
    np.add.at(degree, v, w)
    pi = _normalize_pi(degree)
    q = np.concatenate([w / degree[u], w / degree[v]])
    rev = _build_reverse_edge_map(src, dst)
    return _finalize_graph(num_nodes, src, dst, rev, q, pi, atol=atol, rtol=rtol)


def build_graph_from_directed_rates(
    num_nodes: int,
    src: Iterable[int],
    dst: Iterable[int],
    q: Iterable[float],
    *,
    pi: Iterable[float] | None = None,
    check_reversible: bool = True,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    tol_ratio: float = 1e-10,
) -> GraphSpec:
    """Construct a graph from directed rates and an optional stationary law.

    Args:
        num_nodes: Total number of graph nodes.
        src: Source node for each directed edge.
        dst: Destination node for each directed edge.
        q: Positive directed edge rates.
        pi: Optional stationary distribution. If omitted, it is inferred under
            the reversibility assumption.
        check_reversible: Whether to enforce reversibility and stationarity
            before returning.
        atol: Absolute tolerance for reversibility and stationarity checks.
        rtol: Relative tolerance for reversibility and stationarity checks.
        tol_ratio: Tolerance for log-ratio cycle consistency while inferring
            ``pi``.

    Returns:
        A :class:`GraphSpec` built from the directed rates.

    Raises:
        ValueError: If the rate graph is malformed, disconnected, missing a
            reverse edge, or fails the requested invariants.
    """

    src_arr = _as_int_array(src)
    dst_arr = _as_int_array(dst)
    q_arr = _as_float_array(q)
    if not (src_arr.size == dst_arr.size == q_arr.size):
        raise ValueError("src, dst, and q must have the same length")
    if num_nodes <= 1:
        raise ValueError("num_nodes must be at least 2")
    if np.any((src_arr < 0) | (src_arr >= num_nodes) | (dst_arr < 0) | (dst_arr >= num_nodes)):
        raise ValueError("edge endpoints are out of range")
    if np.any(q_arr <= 0):
        raise ValueError("rates must be strictly positive")

    rev = _build_reverse_edge_map(src_arr, dst_arr)
    _check_connected(num_nodes, src_arr, dst_arr)

    if pi is None:
        if not check_reversible:
            raise ValueError("pi must be supplied when check_reversible is False")
        pi_arr = _infer_pi_from_reversible_rates(
            num_nodes,
            src_arr,
            dst_arr,
            rev,
            q_arr,
            tol_ratio=tol_ratio,
        )
    else:
        pi_arr = _normalize_pi(_as_float_array(pi))
        if pi_arr.size != num_nodes:
            raise ValueError("pi must have length num_nodes")

    if check_reversible:
        return _finalize_graph(
            num_nodes,
            src_arr,
            dst_arr,
            rev,
            q_arr,
            pi_arr,
            atol=atol,
            rtol=rtol,
        )

    out_rate = _out_rate(num_nodes, src_arr, q_arr)
    return GraphSpec(
        num_nodes=num_nodes,
        num_edges=int(src_arr.size),
        src=jnp.asarray(src_arr, dtype=jnp.int32),
        dst=jnp.asarray(dst_arr, dtype=jnp.int32),
        rev=jnp.asarray(rev, dtype=jnp.int32),
        q=jnp.asarray(q_arr, dtype=jnp.float64),
        pi=jnp.asarray(pi_arr, dtype=jnp.float64),
        out_rate=jnp.asarray(out_rate, dtype=jnp.float64),
    )
