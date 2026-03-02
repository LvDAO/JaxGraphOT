from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import GraphSpec

Array = jax.Array


def avg_time(rho: Array) -> Array:
    rho = jnp.asarray(rho)
    return 0.5 * (rho[:-1] + rho[1:])


def grad(graph: GraphSpec, phi: Array) -> Array:
    phi = jnp.asarray(phi)
    return phi[:, graph.src] - phi[:, graph.dst]


def div(graph: GraphSpec, m: Array) -> Array:
    m = jnp.asarray(m)
    contribution = 0.5 * graph.q[None, :] * (m[:, graph.rev] - m)
    out = jnp.zeros((m.shape[0], graph.num_nodes), dtype=m.dtype)
    return out.at[:, graph.src].add(contribution)


def laplace(graph: GraphSpec, phi: Array) -> Array:
    return div(graph, grad(graph, phi))


def continuity_residual(
    graph: GraphSpec,
    rho: Array,
    m: Array,
    rho_a: Array,
    rho_b: Array,
) -> Array:
    rho = jnp.asarray(rho)
    m = jnp.asarray(m)
    rho_a = jnp.asarray(rho_a)
    rho_b = jnp.asarray(rho_b)
    h = 1.0 / m.shape[0]
    delta = rho[1:] - rho[:-1]
    delta = delta.at[0].set(rho[1] - rho_a)
    delta = delta.at[-1].set(rho_b - rho[-2])
    return delta / h + div(graph, m)
