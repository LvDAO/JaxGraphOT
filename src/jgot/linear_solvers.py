"""Linear solvers used by the projection operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .operators import continuity_residual
from .types import GraphSpec

Array = jax.Array


def _project_zero_mean(value: Array) -> Array:
    """Project a space-time field onto the zero-mean gauge subspace."""

    return value - jnp.mean(value)


def _build_ceh_constraint_pullback(
    graph: GraphSpec,
    rho: Array,
    m: Array,
) -> tuple:
    """Build the linearized continuity map and its VJP pullback."""

    zero_boundary = jnp.zeros_like(rho[0])
    zero_rho_int = jnp.zeros_like(rho[1:-1])
    zero_m = jnp.zeros_like(m)

    def constraint_map(drho_int: Array, dm: Array) -> Array:
        drho = jnp.zeros_like(rho)
        drho = drho.at[1:-1].set(drho_int)
        return continuity_residual(graph, drho, dm, zero_boundary, zero_boundary)

    _, pullback = jax.vjp(constraint_map, zero_rho_int, zero_m)
    return constraint_map, pullback


def _solve_tridiagonal_jax(dl: Array, d: Array, du: Array, rhs: Array) -> Array:
    """Solve a batch of tridiagonal systems with vector right-hand sides."""

    dl = jnp.asarray(dl)
    d = jnp.asarray(d)
    du = jnp.asarray(du)
    rhs = jnp.asarray(rhs)
    solution = lax.linalg.tridiagonal_solve(dl, d, du, rhs[..., None])
    return solution[..., 0]


def _solve_tridiagonal_javg_thomas_reference(rhs: Array) -> Array:
    """Reference Thomas solve for the ``Javg`` tridiagonal system."""

    rhs = jnp.asarray(rhs)
    num_steps = rhs.shape[0]
    diag = jnp.full((num_steps,), 1.5, dtype=rhs.dtype)
    diag = diag.at[0].set(1.25)
    diag = diag.at[-1].set(1.25)
    off = jnp.full((num_steps - 1,), 0.25, dtype=rhs.dtype)

    c_prime0 = off[0] / diag[0]
    d_prime0 = rhs[0] / diag[0]

    def forward_body(i: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        c_prime, d_prime = state
        denom = diag[i] - off[i - 1] * c_prime[i - 1]
        next_c = jnp.where(i < num_steps - 1, off[i] / denom, 0.0)
        next_d = (rhs[i] - off[i - 1] * d_prime[i - 1]) / denom
        c_prime = c_prime.at[i].set(next_c)
        d_prime = d_prime.at[i].set(next_d)
        return c_prime, d_prime

    c_prime = jnp.zeros((num_steps,), dtype=rhs.dtype).at[0].set(c_prime0)
    d_prime = jnp.zeros_like(rhs).at[0].set(d_prime0)
    c_prime, d_prime = lax.fori_loop(1, num_steps, forward_body, (c_prime, d_prime))

    lam = jnp.zeros_like(rhs).at[-1].set(d_prime[-1])

    def back_body(i: int, lam_arr: Array) -> Array:
        idx = num_steps - 2 - i
        value = d_prime[idx] - c_prime[idx] * lam_arr[idx + 1]
        return lam_arr.at[idx].set(value)

    lam = lax.fori_loop(0, num_steps - 1, back_body, lam)
    return lam


def _build_javg_tridiagonal_coeffs(num_steps: int, dtype) -> tuple[Array, Array, Array]:
    """Return the tridiagonal coefficients for the ``Javg`` projection."""

    lower = jnp.full((num_steps,), 0.25, dtype=dtype)
    lower = lower.at[0].set(0.0)
    diag = jnp.full((num_steps,), 1.5, dtype=dtype)
    diag = diag.at[0].set(1.25)
    diag = diag.at[-1].set(1.25)
    upper = jnp.full((num_steps,), 0.25, dtype=dtype)
    upper = upper.at[-1].set(0.0)
    return lower, diag, upper


def solve_tridiagonal_javg(rhs: Array, rho: Array, rho_a: Array, rho_b: Array) -> Array:
    """Solve the time-only tridiagonal system used by ``project_javg``.

    Args:
        rhs: Right-hand side with a leading time dimension of length ``N``.
        rho: Current node-density path. Included for interface symmetry with the
            caller.
        rho_a: Initial density. Included for interface symmetry with the caller.
        rho_b: Terminal density. Included for interface symmetry with the
            caller.

    Returns:
        The tridiagonal solve result, interpreted by the caller as the
        Lagrange-multiplier array for the ``Javg`` projection.
    """

    del rho, rho_a, rho_b
    rhs = jnp.asarray(rhs)
    num_steps = rhs.shape[0]
    num_nodes = rhs.shape[1]
    lower, diag, upper = _build_javg_tridiagonal_coeffs(num_steps, rhs.dtype)
    lower = jnp.broadcast_to(lower[None, :], (num_nodes, num_steps))
    diag = jnp.broadcast_to(diag[None, :], (num_nodes, num_steps))
    upper = jnp.broadcast_to(upper[None, :], (num_nodes, num_steps))
    rhs_t = jnp.swapaxes(rhs, 0, 1)
    lam_t = _solve_tridiagonal_jax(lower, diag, upper, rhs_t)
    return jnp.swapaxes(lam_t, 0, 1)


def conjugate_gradient(
    matvec,
    b: Array,
    *,
    max_iters: int,
    tol: float,
    x0: Array | None = None,
    preconditioner=None,
) -> tuple[Array, Array, Array]:
    """Run a small JAX-native conjugate-gradient solve.

    Args:
        matvec: Linear operator applied to an array shaped like ``b``.
        b: Right-hand side.
        max_iters: Maximum number of CG iterations.
        tol: Residual tolerance for early stopping inside the fixed loop.
        x0: Optional warm-start iterate.
        preconditioner: Optional callable applied to the residual before the CG
            update.

    Returns:
        A tuple ``(x, residual, iters_used)`` containing the approximate
        solution, final residual norm, and the number of effective iterations.
    """

    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - matvec(x)
    z = r if preconditioner is None else preconditioner(r)
    p = z
    rz_old = jnp.sum(r * z)
    res0 = jnp.sqrt(jnp.sum(r * r))

    def body(
        _: int,
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        x_curr, r_curr, z_curr, p_curr, rz_curr, res_curr, iters_used, done = state
        ap = matvec(p_curr)
        denom = jnp.sum(p_curr * ap)
        alpha = jnp.where(done, 0.0, rz_curr / jnp.maximum(denom, 1e-30))
        x_next = x_curr + alpha * p_curr
        r_next = r_curr - alpha * ap
        z_next = r_next if preconditioner is None else preconditioner(r_next)
        rz_next = jnp.sum(r_next * z_next)
        res_next = jnp.sqrt(jnp.sum(r_next * r_next))
        done_next = done | (res_next <= tol)
        beta = jnp.where(done_next, 0.0, rz_next / jnp.maximum(rz_curr, 1e-30))
        p_next = z_next + beta * p_curr
        x_final = jnp.where(done, x_curr, x_next)
        r_final = jnp.where(done, r_curr, r_next)
        z_final = jnp.where(done, z_curr, z_next)
        p_final = jnp.where(done_next, jnp.zeros_like(p_curr), p_next)
        rz_final = jnp.where(done, rz_curr, rz_next)
        res_final = jnp.where(done, res_curr, res_next)
        iters_final = jnp.where(done, iters_used, iters_used + 1)
        return x_final, r_final, z_final, p_final, rz_final, res_final, iters_final, done_next

    x, _, _, _, _, residual, iters_used, _ = lax.fori_loop(
        0,
        max_iters,
        body,
        (x, r, z, p, rz_old, res0, jnp.array(0, dtype=jnp.int32), res0 <= tol),
    )
    return x, residual, iters_used


def _build_ceh_matvec(graph: GraphSpec, rho: Array, m: Array):
    """Build the weighted matrix-free ``CE_h`` normal operator."""

    constraint_map, pullback = _build_ceh_constraint_pullback(graph, rho, m)
    pi_weight = graph.pi[None, :]
    edge_weight = 0.5 * graph.q[None, :] * graph.pi[graph.src][None, :]
    rho_inv_weight = 1.0 / jnp.maximum(pi_weight, 1e-30)
    m_inv_weight = 1.0 / jnp.maximum(edge_weight, 1e-30)

    def matvec(phi: Array) -> Array:
        phi = _project_zero_mean(phi)
        drho_int_adj, dm_adj = pullback(phi)
        drho_int = rho_inv_weight * drho_int_adj
        dm = m_inv_weight * dm_adj
        projected = constraint_map(drho_int, dm)
        return _project_zero_mean(projected)

    return matvec


def _build_ceh_rhs(graph: GraphSpec, rho: Array, m: Array, rho_a: Array, rho_b: Array) -> Array:
    """Build the right-hand side for the gauge-fixed ``CE_h`` normal equations."""

    residual = continuity_residual(graph, rho, m, rho_a, rho_b)
    return _project_zero_mean(residual)


def _build_ceh_preconditioner(
    graph: GraphSpec,
    *,
    num_steps: int,
    dtype,
    preconditioner: str,
):
    """Build the requested preconditioner for the ``CE_h`` normal equations."""

    h = 1.0 / num_steps
    pi_inv = 1.0 / jnp.maximum(graph.pi[None, :], 1e-30)
    time_diag = jnp.full((num_steps, 1), 2.0 / (h * h), dtype=dtype)
    time_diag = time_diag.at[0].set(1.0 / (h * h))
    time_diag = time_diag.at[-1].set(1.0 / (h * h))
    diag = jnp.maximum((time_diag + 0.5 * graph.out_rate[None, :]) * pi_inv, 1e-12)

    if preconditioner == "jacobi":
        diag_inv = 1.0 / diag

        def apply(value: Array) -> Array:
            value = _project_zero_mean(value)
            return _project_zero_mean(diag_inv * value)

        return apply

    if preconditioner == "block_jacobi":
        lower_base = jnp.full((num_steps,), -1.0 / (h * h), dtype=dtype)
        lower_base = lower_base.at[0].set(0.0)
        upper_base = jnp.full((num_steps,), -1.0 / (h * h), dtype=dtype)
        upper_base = upper_base.at[-1].set(0.0)
        pi_inv_vec = 1.0 / jnp.maximum(graph.pi, 1e-30)
        lower = pi_inv_vec[:, None] * lower_base[None, :]
        upper = pi_inv_vec[:, None] * upper_base[None, :]
        diag_blocks = diag.T

        def apply(value: Array) -> Array:
            value = _project_zero_mean(value)
            solved_t = _solve_tridiagonal_jax(lower, diag_blocks, upper, jnp.swapaxes(value, 0, 1))
            return _project_zero_mean(jnp.swapaxes(solved_t, 0, 1))

        return apply

    raise ValueError(f"unsupported CE_h preconditioner: {preconditioner}")


def solve_ceh_gauge_fixed(
    graph: GraphSpec,
    rho: Array,
    m: Array,
    rho_a: Array,
    rho_b: Array,
    *,
    cg_max_iters: int,
    cg_tol: float,
    x0: Array | None = None,
    preconditioner: str = "jacobi",
) -> tuple[Array, Array, Array]:
    """Solve the weighted gauge-fixed normal equations for ``CE_h``.

    This variant follows the weighted Hilbert-space structure used by the paper:
    node increments are weighted by ``pi`` and edge increments by
    ``Q(x, y) * pi(x)`` (represented by the sparse edge list). The normal
    equation is applied matrix-free as ``C M^{-1} C^T`` and restricted to the
    zero-mean gauge subspace.
    """

    rho = jnp.asarray(rho)
    m = jnp.asarray(m)
    rho_a = jnp.asarray(rho_a)
    rho_b = jnp.asarray(rho_b)
    num_steps = m.shape[0]
    matvec = _build_ceh_matvec(graph, rho, m)
    b = _build_ceh_rhs(graph, rho, m, rho_a, rho_b)
    precond = _build_ceh_preconditioner(
        graph,
        num_steps=num_steps,
        dtype=rho.dtype,
        preconditioner=preconditioner,
    )

    x_init = None if x0 is None else _project_zero_mean(jnp.asarray(x0))
    phi, residual_norm, iters_used = conjugate_gradient(
        matvec,
        b,
        max_iters=cg_max_iters,
        tol=cg_tol,
        x0=x_init,
        preconditioner=precond,
    )
    return _project_zero_mean(phi), residual_norm, iters_used
