"""Projection and proximal operators used by the PDHG split solve."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .linear_solvers import solve_ceh_gauge_fixed, solve_tridiagonal_javg
from .means import MeanOps
from .operators import avg_time, continuity_residual
from .types import GraphSpec

Array = jax.Array


def project_ceh(
    graph: GraphSpec,
    rho: Array,
    m: Array,
    rho_a: Array,
    rho_b: Array,
    *,
    cg_max_iters: int,
    cg_tol: float,
    phi0: Array | None = None,
    cg_warm_start: bool = True,
    cg_preconditioner: str = "jacobi",
) -> tuple[Array, Array, Array, Array, Array]:
    """Project onto the discrete continuity constraint with fixed endpoints.

    Args:
        graph: Sparse reversible graph.
        rho: Node-density path with shape ``(N + 1, X)``.
        m: Edge flux path with shape ``(N, E)``.
        rho_a: Fixed initial density with shape ``(X,)``.
        rho_b: Fixed terminal density with shape ``(X,)``.
        cg_max_iters: Maximum number of inner CG iterations.
        cg_tol: Residual tolerance for the inner CG solve.
        phi0: Optional warm start for the dual potential.
        cg_warm_start: Whether to use ``phi0``.
        cg_preconditioner: Preconditioner identifier for the inner CG solve.

    Returns:
        A tuple ``(rho_pr, m_pr, phi, cg_residual, cg_iters)`` containing the
        projected state, the dual potential, and CG diagnostics.
    """

    rho = jnp.asarray(rho)
    m = jnp.asarray(m)
    rho_a = jnp.asarray(rho_a)
    rho_b = jnp.asarray(rho_b)
    phi_init = phi0 if cg_warm_start else None
    phi, cg_residual, cg_iters = solve_ceh_gauge_fixed(
        graph,
        rho,
        m,
        rho_a,
        rho_b,
        cg_max_iters=cg_max_iters,
        cg_tol=cg_tol,
        x0=phi_init,
        preconditioner=cg_preconditioner,
    )
    zero_boundary = jnp.zeros_like(rho_a)

    def constraint_map(drho_int: Array, dm: Array) -> Array:
        drho = jnp.zeros_like(rho)
        drho = drho.at[1:-1].set(drho_int)
        return continuity_residual(graph, drho, dm, zero_boundary, zero_boundary)

    _, pullback = jax.vjp(constraint_map, jnp.zeros_like(rho[1:-1]), jnp.zeros_like(m))
    drho_int, dm = pullback(phi)
    drho = jnp.zeros_like(rho).at[1:-1].set(drho_int)
    rho_pr = (rho - drho).at[0].set(rho_a).at[-1].set(rho_b)
    m_pr = m - dm
    return rho_pr, m_pr, phi, cg_residual, cg_iters


def prox_a_star(vartheta: Array, m: Array, *, newton_iters: int) -> tuple[Array, Array]:
    """Apply the pointwise proximal operator for the action conjugate term.

    Args:
        vartheta: Mean-related edge variable with shape ``(N, E)``.
        m: Edge flux variable with shape ``(N, E)``.
        newton_iters: Number of fixed Newton iterations used pointwise.

    Returns:
        The updated ``(vartheta, m)`` pair after the pointwise proximal step.
    """

    vartheta = jnp.asarray(vartheta)
    m = jnp.asarray(m)
    feasible = vartheta + 0.25 * m * m <= 0.0
    v = m

    def body(_: int, value: Array) -> Array:
        f = value**3 + 4.0 * (vartheta + 2.0) * value - 8.0 * m
        df = 3.0 * value * value + 4.0 * (vartheta + 2.0)
        step = f / jnp.where(jnp.abs(df) > 1e-12, df, 1.0)
        return value - step

    v = lax.fori_loop(0, newton_iters, body, v)
    p_proj = -0.25 * v * v
    q_proj = v
    p_proj = jnp.where(feasible, vartheta, p_proj)
    q_proj = jnp.where(feasible, m, q_proj)
    return p_proj, q_proj


def project_jpm(
    graph: GraphSpec,
    q_node: Array,
    rho_minus: Array,
    rho_plus: Array,
) -> tuple[Array, Array, Array]:
    """Project onto the ``J±`` coupling constraint.

    Args:
        graph: Sparse reversible graph.
        q_node: Nodewise auxiliary variable with shape ``(N, X)``.
        rho_minus: Edge-local source densities with shape ``(N, E)``.
        rho_plus: Edge-local destination densities with shape ``(N, E)``.

    Returns:
        A tuple ``(q_node_pr, rho_minus_pr, rho_plus_pr)`` satisfying the
        ``J±`` consistency relation.
    """

    q_node = jnp.asarray(q_node)
    rho_minus = jnp.asarray(rho_minus)
    rho_plus = jnp.asarray(rho_plus)
    term = 0.5 * graph.q[None, :] * (rho_minus + rho_plus[:, graph.rev])
    numer = q_node.at[:, graph.src].add(term)
    rho_pr = numer / (1.0 + graph.out_rate[None, :])
    rho_minus_pr = rho_pr[:, graph.src]
    rho_plus_pr = rho_pr[:, graph.dst]
    return rho_pr, rho_minus_pr, rho_plus_pr


def prox_i_star_jpm(
    graph: GraphSpec,
    q_node: Array,
    rho_minus: Array,
    rho_plus: Array,
) -> tuple[Array, Array, Array]:
    """Apply ``id - project_jpm`` for the dual ``J±`` term."""

    q_pr, rho_minus_pr, rho_plus_pr = project_jpm(graph, q_node, rho_minus, rho_plus)
    return q_node - q_pr, rho_minus - rho_minus_pr, rho_plus - rho_plus_pr


def project_javg(rho: Array, rho_bar: Array, rho_a: Array, rho_b: Array) -> tuple[Array, Array]:
    """Project onto the time-average consistency constraint.

    Args:
        rho: Node-density path with shape ``(N + 1, X)``.
        rho_bar: Time-averaged node densities with shape ``(N, X)``.
        rho_a: Fixed initial density with shape ``(X,)``.
        rho_b: Fixed terminal density with shape ``(X,)``.

    Returns:
        A tuple ``(rho_pr, rho_bar_pr)`` satisfying the ``Javg`` relation.
    """

    rho = jnp.asarray(rho)
    rho_bar = jnp.asarray(rho_bar)
    rho_a = jnp.asarray(rho_a)
    rho_b = jnp.asarray(rho_b)
    rhs = rho_bar - 0.5 * (rho[:-1] + rho[1:])
    rhs = rhs.at[0].set(rho_bar[0] - 0.5 * (rho_a + rho[1]))
    rhs = rhs.at[-1].set(rho_bar[-1] - 0.5 * (rho_b + rho[-2]))
    lam = solve_tridiagonal_javg(rhs, rho, rho_a, rho_b)
    rho_pr = rho.at[0].set(rho_a).at[-1].set(rho_b)
    interior = rho[1:-1] + 0.5 * (lam[:-1] + lam[1:])
    rho_pr = rho_pr.at[1:-1].set(interior)
    rho_bar_pr = rho_bar - lam
    return rho_pr, rho_bar_pr


def prox_i_star_javg(rho: Array, rho_bar: Array, rho_a: Array, rho_b: Array) -> tuple[Array, Array]:
    """Apply ``id - project_javg`` for the dual time-average term."""

    rho_pr, rho_bar_pr = project_javg(rho, rho_bar, rho_a, rho_b)
    return rho - rho_pr, rho_bar - rho_bar_pr


def project_jeq(rho_bar: Array, q_node: Array) -> tuple[Array, Array]:
    """Project onto the equality constraint ``rho_bar == q_node``."""

    rho_bar = jnp.asarray(rho_bar)
    q_node = jnp.asarray(q_node)
    mid = 0.5 * (rho_bar + q_node)
    return mid, mid


def _project_k_point(mean_ops: MeanOps, p1: Array, p2: Array, p3: Array) -> Array:
    """Project one point onto the mean-admissible ``K`` set."""

    theta_val = mean_ops.theta(p1, p2)
    inside = (p1 >= 0) & (p2 >= 0) & (p3 >= 0) & (p3 <= theta_val + 1e-12)

    def inside_branch(_: None) -> Array:
        return jnp.asarray([p1, p2, p3])

    def outside_branch(_: None) -> Array:
        def bottom_branch(_: None) -> Array:
            return jnp.asarray([jnp.maximum(p1, 0.0), jnp.maximum(p2, 0.0), 0.0])

        def positive_c_branch(_: None) -> Array:
            origin_case = (
                (p1 <= 0)
                & (p2 <= 0)
                & mean_ops.origin_supergrad_contains(-p1 / p3, -p2 / p3)
            )

            def origin_branch(_: None) -> Array:
                return jnp.zeros((3,), dtype=jnp.result_type(p1, p2, p3))

            def top_branch(_: None) -> Array:
                q1, q2, q3 = mean_ops.project_k_top(p1, p2, p3)
                return jnp.asarray([q1, q2, q3])

            return lax.cond(origin_case, origin_branch, top_branch, operand=None)

        return lax.cond(p3 <= 0, bottom_branch, positive_c_branch, operand=None)

    return lax.cond(inside, inside_branch, outside_branch, operand=None)


def project_k(
    mean_ops: MeanOps,
    rho_minus: Array,
    rho_plus: Array,
    vartheta: Array,
) -> tuple[Array, Array, Array]:
    """Project edgewise onto the admissible set defined by the mean function.

    Args:
        mean_ops: Mean implementation that defines the admissible set.
        rho_minus: Edge-local source densities with shape ``(N, E)``.
        rho_plus: Edge-local destination densities with shape ``(N, E)``.
        vartheta: Mean-related edge variable with shape ``(N, E)``.

    Returns:
        A tuple ``(rho_minus_pr, rho_plus_pr, vartheta_pr)`` after the
        pointwise ``K`` projection.

    Notes:
        The paper treats this as an exact pointwise Euclidean projection. The
        current implementation uses fixed-iteration scalar root solves through
        :class:`jgot.LogMeanOps`, which is practical for JAX but can introduce
        drift near the singular boundary on larger problems.
    """

    rho_minus = jnp.asarray(rho_minus)
    rho_plus = jnp.asarray(rho_plus)
    vartheta = jnp.asarray(vartheta)
    flat = jax.vmap(lambda a, b, c: _project_k_point(mean_ops, a, b, c))(
        rho_minus.reshape(-1),
        rho_plus.reshape(-1),
        vartheta.reshape(-1),
    )
    proj = flat.reshape(rho_minus.shape + (3,))
    return proj[..., 0], proj[..., 1], proj[..., 2]


def init_split_state(
    graph: GraphSpec,
    rho: Array,
    mean_ops: MeanOps,
) -> tuple[Array, Array, Array, Array, Array]:
    """Derive the auxiliary split variables from a node-density path.

    Args:
        graph: Sparse reversible graph.
        rho: Node-density path with shape ``(N + 1, X)``.
        mean_ops: Mean implementation used to initialize ``vartheta``.

    Returns:
        A tuple ``(rho_bar, q_node, rho_minus, rho_plus, vartheta)`` suitable
        for constructing an :class:`jgot.OTState`.
    """

    rho = jnp.asarray(rho)
    rho_bar = avg_time(rho)
    q_node = rho_bar
    rho_minus = q_node[:, graph.src]
    rho_plus = q_node[:, graph.dst]
    vartheta = mean_ops.theta(rho_minus, rho_plus)
    return rho_bar, q_node, rho_minus, rho_plus, vartheta
