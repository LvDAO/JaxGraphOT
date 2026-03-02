"""Discrete graph and time operators used throughout the solver."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import GraphSpec

Array = jax.Array


def avg_time(rho: Array) -> Array:
    """Average consecutive time slices of a node path.

    Args:
        rho: Node densities with shape ``(N + 1, X)``.

    Returns:
        Time-averaged densities with shape ``(N, X)``.
    """

    rho = jnp.asarray(rho)
    return 0.5 * (rho[:-1] + rho[1:])


def grad(graph: GraphSpec, phi: Array) -> Array:
    """Apply the directed-edge gradient operator.

    Args:
        graph: Sparse reversible graph.
        phi: Node field with shape ``(T, X)``.

    Returns:
        Edge field with shape ``(T, E)`` computed as
        ``phi[:, src] - phi[:, dst]``.
    """

    phi = jnp.asarray(phi)
    return phi[:, graph.src] - phi[:, graph.dst]


def div(graph: GraphSpec, m: Array) -> Array:
    """Apply the directed-edge divergence operator.

    Args:
        graph: Sparse reversible graph.
        m: Edge field with shape ``(T, E)``.

    Returns:
        Node field with shape ``(T, X)``. This is the divergence convention
        used by the solver's discrete continuity equation.
    """

    m = jnp.asarray(m)
    contribution = 0.5 * graph.q[None, :] * (m[:, graph.rev] - m)
    out = jnp.zeros((m.shape[0], graph.num_nodes), dtype=m.dtype)
    return out.at[:, graph.src].add(contribution)


def laplace(graph: GraphSpec, phi: Array) -> Array:
    """Apply the graph Laplacian induced by ``div(grad(phi))``."""

    return div(graph, grad(graph, phi))


def continuity_residual(
    graph: GraphSpec,
    rho: Array,
    m: Array,
    rho_a: Array,
    rho_b: Array,
) -> Array:
    """Return the discrete continuity-equation residual with fixed endpoints.

    Args:
        graph: Sparse reversible graph.
        rho: Node-density path with shape ``(N + 1, X)``.
        m: Edge flux path with shape ``(N, E)``.
        rho_a: Fixed initial density with shape ``(X,)``.
        rho_b: Fixed terminal density with shape ``(X,)``.

    Returns:
        Residual array with shape ``(N, X)``. Exact feasibility corresponds to
        a residual of zero.
    """

    rho = jnp.asarray(rho)
    m = jnp.asarray(m)
    rho_a = jnp.asarray(rho_a)
    rho_b = jnp.asarray(rho_b)
    h = 1.0 / m.shape[0]
    delta = rho[1:] - rho[:-1]
    delta = delta.at[0].set(rho[1] - rho_a)
    delta = delta.at[-1].set(rho_b - rho[-2])
    return delta / h + div(graph, m)
