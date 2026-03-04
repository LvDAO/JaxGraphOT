"""Top-level solve entrypoint and user-facing solver orchestration."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .pdhg import initialize_state, run_pdhg
from .projections import init_split_state, project_ceh
from .types import OTConfig, OTDebugTrace, OTProblem, OTSolution, OTState

Array = jax.Array


def _validate_density(name: str, rho: Array, pi: Array, num_nodes: int) -> None:
    """Validate one endpoint density against the graph normalization rules."""

    rho = jnp.asarray(rho, dtype=jnp.float64)
    if rho.ndim != 1 or rho.shape[0] != num_nodes:
        raise ValueError(f"{name} must have shape ({num_nodes},)")
    if bool(jnp.any(rho < -1e-12)):
        raise ValueError(f"{name} must be nonnegative")
    mass = float(jnp.sum(pi * rho))
    if abs(mass - 1.0) > 1e-8:
        raise ValueError(f"{name} must satisfy sum(pi * rho) == 1")


def compute_action(problem: OTProblem, state: OTState) -> Array:
    """Compute the discrete transport action for a solved state.

    Args:
        problem: Problem definition that supplies the graph and time step.
        state: Split state returned by the solver.

    Returns:
        The discrete action value. The reported transport distance is
        ``sqrt(action)``.
    """

    h = problem.time.h
    weights = 0.5 * h * problem.graph.q[None, :] * problem.graph.pi[problem.graph.src][None, :]
    safe = jnp.where(
        state.vartheta > 0,
        (state.m * state.m) / state.vartheta,
        jnp.where(jnp.abs(state.m) <= 1e-12, 0.0, jnp.inf),
    )
    return jnp.sum(weights * safe)


def _build_trivial_state(problem: OTProblem, rho: Array) -> OTState:
    """Build the exact zero-flow state used when the endpoints already match."""

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
    """Construct the default feasible warm start used before PDHG iterations."""

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


def _wrap_debug_trace(trace_payload: dict[str, Array]) -> OTDebugTrace:
    """Convert raw JAX checkpoint buffers into the public debug-trace type."""

    return OTDebugTrace(
        iterations=trace_payload["iterations"],
        action=trace_payload["action"],
        continuity_residual=trace_payload["continuity_residual"],
        primal_delta=trace_payload["primal_delta"],
        dual_delta=trace_payload["dual_delta"],
        max_constraint_residual=trace_payload["max_constraint_residual"],
        ceh_cg_residual=trace_payload["ceh_cg_residual"],
        ceh_cg_iters=trace_payload["ceh_cg_iters"],
        min_vartheta=trace_payload["min_vartheta"],
        num_records=int(trace_payload["num_records"]),
    )


def solve_ot(problem: OTProblem, config: OTConfig = OTConfig()) -> OTSolution:
    """Solve the two-endpoint dynamic OT problem on a sparse reversible graph.

    This is the main public entrypoint. The function validates the endpoint
    densities, checks that each has shape ``(graph.num_nodes,)``, enforces
    nonnegativity, and requires ``sum(graph.pi * rho) == 1`` for both
    endpoints. If the endpoints already match, it returns the exact zero-action
    shortcut. Otherwise, it runs the JAX/JIT-backed PDHG solver and returns the
    resulting time-discrete geodesic state.

    Args:
        problem: Fully specified OT problem instance.
        config: Optional solver configuration. Defaults are the intended
            starting point for normal usage.

    Returns:
        An :class:`OTSolution` containing the transport distance, discrete
        action, state trajectory, iteration count, convergence flag, and
        diagnostics.

    Raises:
        ValueError: If either endpoint density has the wrong shape, contains
            negative entries, or fails the ``sum(pi * rho) == 1`` normalization
            rule.
    """

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
        debug_trace = None
        if config.record_debug_trace:
            debug_trace = OTDebugTrace(
                iterations=jnp.array([1], dtype=jnp.int32),
                action=jnp.array([0.0], dtype=rho_a.dtype),
                continuity_residual=jnp.array([0.0], dtype=rho_a.dtype),
                primal_delta=jnp.array([0.0], dtype=rho_a.dtype),
                dual_delta=jnp.array([0.0], dtype=rho_a.dtype),
                max_constraint_residual=jnp.array([0.0], dtype=rho_a.dtype),
                ceh_cg_residual=jnp.array([0.0], dtype=rho_a.dtype),
                ceh_cg_iters=jnp.array([0], dtype=jnp.int32),
                min_vartheta=jnp.array([jnp.min(state.vartheta)], dtype=rho_a.dtype),
                num_records=1,
            )
        return OTSolution(
            distance=zero,
            action=zero,
            state=state,
            iterations_used=1,
            converged=True,
            diagnostics=diagnostics,
            debug_trace=debug_trace,
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
        state, diagnostics, iterations_used, converged, trace_payload = run_pdhg(
            problem.graph,
            rho_a,
            rho_b,
            problem.mean_ops,
            problem.time.num_steps,
            config,
            initial_primal=primal0,
        )
        action = compute_action(problem, state)
        return state, diagnostics, iterations_used, converged, action, trace_payload

    state, diagnostics, iterations_used, converged, action, trace_payload = _solve_kernel(
        initial_primal
    )
    distance = jnp.sqrt(jnp.maximum(action, 0.0))
    converged_flag = bool(
        converged
        & jnp.isfinite(action)
        & (diagnostics["max_constraint_residual"] <= config.feasibility_tol)
    )
    debug_trace = _wrap_debug_trace(trace_payload) if config.record_debug_trace else None
    return OTSolution(
        distance=distance,
        action=action,
        state=state,
        iterations_used=int(iterations_used),
        converged=converged_flag,
        diagnostics=diagnostics,
        debug_trace=debug_trace,
    )
