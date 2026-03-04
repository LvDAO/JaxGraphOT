"""PDHG iteration for the split dynamic OT problem."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from .means import MeanOps
from .operators import continuity_residual
from .projections import (
    init_split_state,
    project_ceh,
    project_jeq,
    project_k,
    prox_a_star,
    prox_i_star_javg,
    prox_i_star_jpm,
)
from .types import GraphSpec, OTConfig, OTState

Array = jax.Array


def _tree_map(fn, *trees):
    """Apply ``jax.tree_map`` to one or more matching pytrees."""

    return jax.tree_util.tree_map(fn, *trees)


def _state_add(left: OTState, right: OTState) -> OTState:
    """Add two split states blockwise."""

    return _tree_map(lambda a, b: a + b, left, right)


def _state_sub(left: OTState, right: OTState) -> OTState:
    """Subtract two split states blockwise."""

    return _tree_map(lambda a, b: a - b, left, right)


def _state_scale(state: OTState, scale: float | Array) -> OTState:
    """Scale every block in a split state by a scalar."""

    return _tree_map(lambda x: scale * x, state)


def _state_norm(state: OTState) -> Array:
    """Compute the Euclidean norm across all split-state blocks."""

    return jnp.sqrt(sum(jnp.sum(arr * arr) for arr in jax.tree_util.tree_leaves(state)))


def _compute_action_from_state(graph: GraphSpec, h: float, state: OTState) -> Array:
    """Compute the discrete action directly from a split state.

    This mirrors :func:`jgot.solver.compute_action` so the PDHG loop can record
    checkpointed action values without leaving the JIT-compiled path.
    """

    weights = 0.5 * h * graph.q[None, :] * graph.pi[graph.src][None, :]
    safe = jnp.where(
        state.vartheta > 0,
        (state.m * state.m) / state.vartheta,
        jnp.where(jnp.abs(state.m) <= 1e-12, 0.0, jnp.inf),
    )
    return jnp.sum(weights * safe)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PDHGCarry:
    """Loop carry for the JAX ``while_loop`` PDHG iteration.

    Attributes:
        primal: Current primal split state.
        dual: Current dual split state.
        primal_bar: Over-relaxed primal state used by PDHG.
        phi_cache: Cached dual potential for warm-starting ``project_ceh``.
        iterations_used: Number of completed PDHG iterations.
        converged: Boolean-valued JAX array tracking the stopping rule.
        diagnostics: Latest residual and inner-solver diagnostics.
        history_*: Fixed-size checkpoint buffers used for optional debug traces.
    """

    primal: OTState
    dual: OTState
    primal_bar: OTState
    phi_cache: Array
    iterations_used: Array
    converged: Array
    diagnostics: dict[str, Array]
    history_iterations: Array
    history_action: Array
    history_continuity: Array
    history_primal_delta: Array
    history_dual_delta: Array
    history_max_constraint: Array
    history_ceh_cg_residual: Array
    history_ceh_cg_iters: Array
    history_min_vartheta: Array
    history_count: Array

    def tree_flatten(self):
        """Return the JAX pytree children for the PDHG loop carry."""

        children = (
            self.primal,
            self.dual,
            self.primal_bar,
            self.phi_cache,
            self.iterations_used,
            self.converged,
            self.diagnostics,
            self.history_iterations,
            self.history_action,
            self.history_continuity,
            self.history_primal_delta,
            self.history_dual_delta,
            self.history_max_constraint,
            self.history_ceh_cg_residual,
            self.history_ceh_cg_iters,
            self.history_min_vartheta,
            self.history_count,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct the PDHG loop carry from JAX pytree children."""

        del aux_data
        return cls(*children)


def initialize_state(
    graph: GraphSpec,
    rho_a: Array,
    rho_b: Array,
    mean_ops: MeanOps,
    num_steps: int,
) -> OTState:
    """Initialize the primal state by linear interpolation in node densities.

    Args:
        graph: Sparse reversible graph.
        rho_a: Initial density with shape ``(X,)``.
        rho_b: Terminal density with shape ``(X,)``.
        mean_ops: Mean implementation used for the auxiliary variables.
        num_steps: Number of time intervals ``N``.

    Returns:
        An :class:`OTState` with linearly interpolated ``rho`` and zero initial
        edge flux.
    """

    alpha = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=rho_a.dtype)[:, None]
    rho = (1.0 - alpha) * rho_a[None, :] + alpha * rho_b[None, :]
    rho_bar, q_node, rho_minus, rho_plus, vartheta = init_split_state(graph, rho, mean_ops)
    m = jnp.zeros((num_steps, graph.num_edges), dtype=rho.dtype)
    return OTState(
        rho=rho,
        m=m,
        vartheta=vartheta,
        rho_minus=rho_minus,
        rho_plus=rho_plus,
        rho_bar=rho_bar,
        q_node=q_node,
    )


def prox_f_star(
    state: OTState,
    graph: GraphSpec,
    rho_a: Array,
    rho_b: Array,
    config: OTConfig,
) -> OTState:
    """Apply the dual proximal step for the ``F*`` split term."""

    vartheta, m = prox_a_star(state.vartheta, state.m, newton_iters=config.newton_iters)
    q_node, rho_minus, rho_plus = prox_i_star_jpm(
        graph,
        state.q_node,
        state.rho_minus,
        state.rho_plus,
    )
    rho, rho_bar = prox_i_star_javg(state.rho, state.rho_bar, rho_a, rho_b)
    return OTState(
        rho=rho,
        m=m,
        vartheta=vartheta,
        rho_minus=rho_minus,
        rho_plus=rho_plus,
        rho_bar=rho_bar,
        q_node=q_node,
    )


def prox_g(
    state: OTState,
    graph: GraphSpec,
    rho_a: Array,
    rho_b: Array,
    mean_ops: MeanOps,
    config: OTConfig,
    phi0: Array | None = None,
) -> tuple[OTState, Array, dict[str, Array]]:
    """Apply the primal projection step for the ``G`` split term.

    Returns:
        A tuple ``(state, phi, ceh_stats)`` containing the updated primal state,
        the cached ``CE_h`` dual potential, and the associated CG diagnostics.
    """

    rho, m, phi, cg_residual, cg_iters = project_ceh(
        graph,
        state.rho,
        state.m,
        rho_a,
        rho_b,
        cg_max_iters=config.cg_max_iters,
        cg_tol=config.cg_tol,
        phi0=phi0,
        cg_warm_start=config.cg_warm_start,
        cg_preconditioner=config.cg_preconditioner,
    )
    rho_minus, rho_plus, vartheta = project_k(
        mean_ops,
        state.rho_minus,
        state.rho_plus,
        state.vartheta,
    )
    rho_bar, q_node = project_jeq(state.rho_bar, state.q_node)
    next_state = OTState(
        rho=rho,
        m=m,
        vartheta=vartheta,
        rho_minus=rho_minus,
        rho_plus=rho_plus,
        rho_bar=rho_bar,
        q_node=q_node,
    )
    return next_state, phi, {"ceh_cg_residual": cg_residual, "ceh_cg_iters": cg_iters}


def compute_diagnostics(
    primal: OTState,
    prev_primal: OTState,
    dual: OTState,
    prev_dual: OTState,
    graph: GraphSpec,
    rho_a: Array,
    rho_b: Array,
    mean_ops: MeanOps,
    ceh_stats: dict[str, Array],
) -> dict[str, Array]:
    """Compute the residual metrics used by the PDHG stopping rule.

    Returns:
        A dictionary containing primal and dual deltas, continuity residual,
        ``K``-set violation, endpoint residual, the combined constraint
        residual, and the latest ``CE_h`` CG diagnostics.
    """

    primal_delta = _state_norm(_state_sub(primal, prev_primal)) / (1.0 + _state_norm(prev_primal))
    dual_num = jnp.sqrt(
        jnp.sum((dual.m - prev_dual.m) ** 2) + jnp.sum((dual.vartheta - prev_dual.vartheta) ** 2)
    )
    dual_den = jnp.sqrt(jnp.sum(prev_dual.m**2) + jnp.sum(prev_dual.vartheta**2))
    dual_delta = dual_num / (1.0 + dual_den)
    cont = continuity_residual(graph, primal.rho, primal.m, rho_a, rho_b)
    continuity_max = jnp.max(jnp.abs(cont))
    upper_slack = jnp.maximum(
        primal.vartheta - mean_ops.theta(primal.rho_minus, primal.rho_plus),
        0.0,
    )
    lower_slack = jnp.maximum(-primal.vartheta, 0.0)
    negativity = jnp.maximum(
        jnp.maximum(-primal.rho_minus, 0.0),
        jnp.maximum(-primal.rho_plus, 0.0),
    )
    k_violation = jnp.max(jnp.maximum(jnp.maximum(upper_slack, lower_slack), negativity))
    endpoint_residual = jnp.maximum(
        jnp.max(jnp.abs(primal.rho[0] - rho_a)),
        jnp.max(jnp.abs(primal.rho[-1] - rho_b)),
    )
    max_constraint_residual = jnp.maximum(
        jnp.maximum(continuity_max, k_violation),
        endpoint_residual,
    )
    return {
        "primal_delta": primal_delta,
        "dual_delta": dual_delta,
        "continuity_residual": continuity_max,
        "k_violation": k_violation,
        "endpoint_residual": endpoint_residual,
        "max_constraint_residual": max_constraint_residual,
        "ceh_cg_residual": ceh_stats["ceh_cg_residual"],
        "ceh_cg_iters": ceh_stats["ceh_cg_iters"],
    }


def _zero_state_like(state: OTState) -> OTState:
    """Allocate a zero-valued split state with the same shapes as ``state``."""

    return OTState(
        rho=jnp.zeros_like(state.rho),
        m=jnp.zeros_like(state.m),
        vartheta=jnp.zeros_like(state.vartheta),
        rho_minus=jnp.zeros_like(state.rho_minus),
        rho_plus=jnp.zeros_like(state.rho_plus),
        rho_bar=jnp.zeros_like(state.rho_bar),
        q_node=jnp.zeros_like(state.q_node),
    )


def run_pdhg(
    graph: GraphSpec,
    rho_a: Array,
    rho_b: Array,
    mean_ops: MeanOps,
    num_steps: int,
    config: OTConfig,
    *,
    initial_primal: OTState | None = None,
) -> tuple[OTState, dict[str, Array], Array, Array, dict[str, Array]]:
    """Run the PDHG iteration for the split dynamic OT problem.

    Args:
        graph: Sparse reversible graph.
        rho_a: Initial density with shape ``(X,)``.
        rho_b: Terminal density with shape ``(X,)``.
        mean_ops: Mean implementation used in the ``K`` projection.
        num_steps: Number of time intervals ``N``.
        config: Solver configuration, including stopping thresholds.
        initial_primal: Optional warm-start primal state. If omitted, the
            routine uses :func:`initialize_state`.

    Returns:
        A tuple ``(state, diagnostics, iterations_used, converged,
        trace_payload)`` containing the final primal state, latest diagnostics,
        the exact iteration count, the JAX boolean convergence flag, and the
        raw fixed-size checkpoint trace buffers.
    """

    if initial_primal is None:
        init = initialize_state(graph, rho_a, rho_b, mean_ops, num_steps)
    else:
        init = initial_primal
    dual0 = _zero_state_like(init)
    trace_length = (config.max_iters + config.check_every - 1) // config.check_every
    history_dtype = rho_a.dtype
    diagnostics0 = {
        "primal_delta": jnp.array(jnp.inf, dtype=rho_a.dtype),
        "dual_delta": jnp.array(jnp.inf, dtype=rho_a.dtype),
        "continuity_residual": jnp.array(jnp.inf, dtype=rho_a.dtype),
        "k_violation": jnp.array(jnp.inf, dtype=rho_a.dtype),
        "endpoint_residual": jnp.array(jnp.inf, dtype=rho_a.dtype),
        "max_constraint_residual": jnp.array(jnp.inf, dtype=rho_a.dtype),
        "ceh_cg_residual": jnp.array(jnp.inf, dtype=rho_a.dtype),
        "ceh_cg_iters": jnp.array(0, dtype=jnp.int32),
    }
    carry = PDHGCarry(
        primal=init,
        dual=dual0,
        primal_bar=init,
        phi_cache=jnp.zeros((num_steps, graph.num_nodes), dtype=rho_a.dtype),
        iterations_used=jnp.array(0, dtype=jnp.int32),
        converged=jnp.array(False),
        diagnostics=diagnostics0,
        history_iterations=jnp.zeros((trace_length,), dtype=jnp.int32),
        history_action=jnp.zeros((trace_length,), dtype=history_dtype),
        history_continuity=jnp.zeros((trace_length,), dtype=history_dtype),
        history_primal_delta=jnp.zeros((trace_length,), dtype=history_dtype),
        history_dual_delta=jnp.zeros((trace_length,), dtype=history_dtype),
        history_max_constraint=jnp.zeros((trace_length,), dtype=history_dtype),
        history_ceh_cg_residual=jnp.zeros((trace_length,), dtype=history_dtype),
        history_ceh_cg_iters=jnp.zeros((trace_length,), dtype=jnp.int32),
        history_min_vartheta=jnp.zeros((trace_length,), dtype=history_dtype),
        history_count=jnp.array(0, dtype=jnp.int32),
    )
    h = 1.0 / float(num_steps)

    def cond_fn(loop_carry: PDHGCarry) -> Array:
        return (~loop_carry.converged) & (loop_carry.iterations_used < config.max_iters)

    def body_fn(loop_carry: PDHGCarry) -> PDHGCarry:
        dual_trial = _state_add(loop_carry.dual, _state_scale(loop_carry.primal_bar, config.sigma))
        dual_next = prox_f_star(dual_trial, graph, rho_a, rho_b, config)
        primal_trial = _state_sub(loop_carry.primal, _state_scale(dual_next, config.tau))
        primal_next, phi_next, ceh_stats = prox_g(
            primal_trial,
            graph,
            rho_a,
            rho_b,
            mean_ops,
            config,
            phi0=loop_carry.phi_cache,
        )
        primal_bar_next = _state_add(
            primal_next,
            _state_scale(_state_sub(primal_next, loop_carry.primal), config.relaxation),
        )
        diagnostics = compute_diagnostics(
            primal_next,
            loop_carry.primal,
            dual_next,
            loop_carry.dual,
            graph,
            rho_a,
            rho_b,
            mean_ops,
            ceh_stats,
        )
        next_iter = loop_carry.iterations_used + 1
        should_check = ((next_iter % config.check_every) == 0) | (next_iter == config.max_iters)
        # The paper's Chambolle-Pock splitting requires tau * sigma < 1. The
        # current defaults satisfy that but are intentionally aggressive.
        converged = should_check & (
            (diagnostics["primal_delta"] <= config.residual_tol)
            & (diagnostics["dual_delta"] <= config.residual_tol)
            & (diagnostics["max_constraint_residual"] <= config.feasibility_tol)
        )
        next_carry = PDHGCarry(
            primal=primal_next,
            dual=dual_next,
            primal_bar=primal_bar_next,
            phi_cache=phi_next,
            iterations_used=next_iter,
            converged=converged,
            diagnostics=diagnostics,
            history_iterations=loop_carry.history_iterations,
            history_action=loop_carry.history_action,
            history_continuity=loop_carry.history_continuity,
            history_primal_delta=loop_carry.history_primal_delta,
            history_dual_delta=loop_carry.history_dual_delta,
            history_max_constraint=loop_carry.history_max_constraint,
            history_ceh_cg_residual=loop_carry.history_ceh_cg_residual,
            history_ceh_cg_iters=loop_carry.history_ceh_cg_iters,
            history_min_vartheta=loop_carry.history_min_vartheta,
            history_count=loop_carry.history_count,
        )
        should_record = should_check & jnp.asarray(config.record_debug_trace)

        def record_trace(carry: PDHGCarry) -> PDHGCarry:
            slot = carry.history_count
            action_value = _compute_action_from_state(graph, h, primal_next)
            min_vartheta = jnp.min(primal_next.vartheta)
            # When continuity is already small but action becomes non-finite,
            # the current iterate is typically singular on the K/action side
            # rather than failing the CE_h solve.
            return PDHGCarry(
                primal=carry.primal,
                dual=carry.dual,
                primal_bar=carry.primal_bar,
                phi_cache=carry.phi_cache,
                iterations_used=carry.iterations_used,
                converged=carry.converged,
                diagnostics=carry.diagnostics,
                history_iterations=carry.history_iterations.at[slot].set(next_iter),
                history_action=carry.history_action.at[slot].set(action_value),
                history_continuity=carry.history_continuity.at[slot].set(
                    diagnostics["continuity_residual"]
                ),
                history_primal_delta=carry.history_primal_delta.at[slot].set(
                    diagnostics["primal_delta"]
                ),
                history_dual_delta=carry.history_dual_delta.at[slot].set(
                    diagnostics["dual_delta"]
                ),
                history_max_constraint=carry.history_max_constraint.at[slot].set(
                    diagnostics["max_constraint_residual"]
                ),
                history_ceh_cg_residual=carry.history_ceh_cg_residual.at[slot].set(
                    diagnostics["ceh_cg_residual"]
                ),
                history_ceh_cg_iters=carry.history_ceh_cg_iters.at[slot].set(
                    diagnostics["ceh_cg_iters"]
                ),
                history_min_vartheta=carry.history_min_vartheta.at[slot].set(min_vartheta),
                history_count=carry.history_count + 1,
            )

        return lax.cond(should_record, record_trace, lambda c: c, next_carry)

    carry = lax.while_loop(cond_fn, body_fn, carry)
    trace_payload = {
        "iterations": carry.history_iterations,
        "action": carry.history_action,
        "continuity_residual": carry.history_continuity,
        "primal_delta": carry.history_primal_delta,
        "dual_delta": carry.history_dual_delta,
        "max_constraint_residual": carry.history_max_constraint,
        "ceh_cg_residual": carry.history_ceh_cg_residual,
        "ceh_cg_iters": carry.history_ceh_cg_iters,
        "min_vartheta": carry.history_min_vartheta,
        "num_records": carry.history_count,
    }
    return carry.primal, carry.diagnostics, carry.iterations_used, carry.converged, trace_payload
