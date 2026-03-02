from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .operators import continuity_residual
from .types import GraphSpec

Array = jax.Array


def solve_tridiagonal_javg(rhs: Array, rho: Array, rho_a: Array, rho_b: Array) -> Array:
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


def conjugate_gradient(
    matvec,
    b: Array,
    *,
    max_iters: int,
    tol: float,
    x0: Array | None = None,
    preconditioner=None,
) -> tuple[Array, Array, Array]:
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
    rho = jnp.asarray(rho)
    m = jnp.asarray(m)
    rho_a = jnp.asarray(rho_a)
    rho_b = jnp.asarray(rho_b)
    num_steps = m.shape[0]
    h = 1.0 / num_steps
    zero_boundary = jnp.zeros_like(rho_a)
    zero_rho_int = jnp.zeros_like(rho[1:-1])
    zero_m = jnp.zeros_like(m)

    def constraint_map(drho_int: Array, dm: Array) -> Array:
        drho = jnp.zeros_like(rho)
        drho = drho.at[1:-1].set(drho_int)
        return continuity_residual(graph, drho, dm, zero_boundary, zero_boundary)

    _, pullback = jax.vjp(constraint_map, zero_rho_int, zero_m)
    residual = continuity_residual(graph, rho, m, rho_a, rho_b)
    b = residual

    def matvec(phi: Array) -> Array:
        drho_int_adj, dm_adj = pullback(phi)
        projected = constraint_map(drho_int_adj, dm_adj)
        const_mode = jnp.mean(phi) * jnp.ones_like(phi)
        return projected + const_mode

    if preconditioner == "jacobi":
        time_diag = jnp.full((num_steps, 1), 2.0 / (h * h), dtype=rho.dtype)
        time_diag = time_diag.at[0].set(1.0 / (h * h))
        time_diag = time_diag.at[-1].set(1.0 / (h * h))
        spatial_diag = 1.0 + 0.5 * graph.out_rate[None, :]
        gauge_diag = jnp.array(1.0 / (num_steps * graph.num_nodes), dtype=rho.dtype)
        diag = time_diag + spatial_diag + gauge_diag
        diag_inv = 1.0 / jnp.maximum(diag, 1e-30)

        def precond(value: Array) -> Array:
            return diag_inv * value

    else:
        precond = None

    return conjugate_gradient(
        matvec,
        b,
        max_iters=cg_max_iters,
        tol=cg_tol,
        x0=x0,
        preconditioner=precond,
    )
