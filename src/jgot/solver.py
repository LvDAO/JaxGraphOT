from __future__ import annotations

import jax
import jax.numpy as jnp

from .pdhg import initialize_state, run_pdhg
from .projections import init_split_state, project_ceh
from .types import OTConfig, OTProblem, OTSolution, OTState

Array = jax.Array


def _validate_density(name: str, rho: Array, pi: Array, num_nodes: int) -> None:
    rho = jnp.asarray(rho, dtype=jnp.float64)
    if rho.ndim != 1 or rho.shape[0] != num_nodes:
        raise ValueError(f"{name} must have shape ({num_nodes},)")
    if bool(jnp.any(rho < -1e-12)):
        raise ValueError(f"{name} must be nonnegative")
    mass = float(jnp.sum(pi * rho))
    if abs(mass - 1.0) > 1e-8:
        raise ValueError(f"{name} must satisfy sum(pi * rho) == 1")


def compute_action(problem: OTProblem, state: OTState) -> Array:
    h = problem.time.h
    weights = 0.5 * h * problem.graph.q[None, :] * problem.graph.pi[problem.graph.src][None, :]
    safe = jnp.where(
        state.vartheta > 0,
        (state.m * state.m) / state.vartheta,
        jnp.where(jnp.abs(state.m) <= 1e-12, 0.0, jnp.inf),
    )
    return jnp.sum(weights * safe)


def _build_trivial_state(problem: OTProblem, rho: Array) -> OTState:
    num_steps = problem.time.num_steps
    rho_path = jnp.repeat(rho[None, :], num_steps + 1, axis=0)
    rho_bar = 0.5 * (rho_path[:-1] + rho_path[1:])
    q_node = rho_bar
    rho_minus = q_node[:, problem.graph.src]
    rho_plus = q_node[:, problem.graph.dst]
    vartheta = problem.mean_ops.theta(rho_minus, rho_plus)
    m = jnp.zeros((num_steps, problem.graph.num_edges), dtype=rho.dtype)
    return OTState(
        rho=rho_path,
        m=m,
        vartheta=vartheta,
        rho_minus=rho_minus,
        rho_plus=rho_plus,
        rho_bar=rho_bar,
        q_node=q_node,
    )


def _build_linear_warm_start(problem: OTProblem, config: OTConfig) -> OTState:
    base = initialize_state(
        problem.graph,
        jnp.asarray(problem.rho_a, dtype=jnp.float64),
        jnp.asarray(problem.rho_b, dtype=jnp.float64),
        problem.mean_ops,
        problem.time.num_steps,
    )
    rho, m, _, _, _ = project_ceh(
        problem.graph,
        base.rho,
        base.m,
        jnp.asarray(problem.rho_a, dtype=jnp.float64),
        jnp.asarray(problem.rho_b, dtype=jnp.float64),
        cg_max_iters=config.cg_max_iters,
        cg_tol=config.cg_tol,
        phi0=None,
        cg_warm_start=False,
        cg_preconditioner=config.cg_preconditioner,
    )
    rho_bar, q_node, rho_minus, rho_plus, vartheta = init_split_state(
        problem.graph,
        rho,
        problem.mean_ops,
    )
    return OTState(
        rho=rho,
        m=m,
        vartheta=vartheta,
        rho_minus=rho_minus,
        rho_plus=rho_plus,
        rho_bar=rho_bar,
        q_node=q_node,
    )


def solve_ot(problem: OTProblem, config: OTConfig = OTConfig()) -> OTSolution:
    _validate_density("rho_a", problem.rho_a, problem.graph.pi, problem.graph.num_nodes)
    _validate_density("rho_b", problem.rho_b, problem.graph.pi, problem.graph.num_nodes)

    rho_a = jnp.asarray(problem.rho_a, dtype=jnp.float64)
    rho_b = jnp.asarray(problem.rho_b, dtype=jnp.float64)

    if bool(jnp.max(jnp.abs(rho_a - rho_b)) <= 1e-12):
        state = _build_trivial_state(problem, rho_a)
        zero = jnp.array(0.0, dtype=rho_a.dtype)
        diagnostics = {
            "primal_delta": zero,
            "dual_delta": zero,
            "continuity_residual": zero,
            "k_violation": zero,
            "endpoint_residual": zero,
            "max_constraint_residual": zero,
            "ceh_cg_residual": zero,
            "ceh_cg_iters": jnp.array(0, dtype=jnp.int32),
        }
        return OTSolution(
            distance=zero,
            action=zero,
            state=state,
            iterations_used=1,
            converged=True,
            diagnostics=diagnostics,
        )

    if config.warm_start == "linear_path":
        initial_primal = _build_linear_warm_start(problem, config)
    else:
        initial_primal = initialize_state(
            problem.graph,
            rho_a,
            rho_b,
            problem.mean_ops,
            problem.time.num_steps,
        )

    @jax.jit
    def _solve_kernel(primal0: OTState):
        state, diagnostics, iterations_used, converged = run_pdhg(
            problem.graph,
            rho_a,
            rho_b,
            problem.mean_ops,
            problem.time.num_steps,
            config,
            initial_primal=primal0,
        )
        action = compute_action(problem, state)
        return state, diagnostics, iterations_used, converged, action

    state, diagnostics, iterations_used, converged, action = _solve_kernel(initial_primal)
    distance = jnp.sqrt(jnp.maximum(action, 0.0))
    converged_flag = bool(
        converged
        & jnp.isfinite(action)
        & (diagnostics["max_constraint_residual"] <= config.feasibility_tol)
    )
    return OTSolution(
        distance=distance,
        action=action,
        state=state,
        iterations_used=int(iterations_used),
        converged=converged_flag,
        diagnostics=diagnostics,
    )
