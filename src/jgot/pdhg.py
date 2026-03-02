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
    return jax.tree_util.tree_map(fn, *trees)


def _state_add(left: OTState, right: OTState) -> OTState:
    return _tree_map(lambda a, b: a + b, left, right)


def _state_sub(left: OTState, right: OTState) -> OTState:
    return _tree_map(lambda a, b: a - b, left, right)


def _state_scale(state: OTState, scale: float | Array) -> OTState:
    return _tree_map(lambda x: scale * x, state)


def _state_norm(state: OTState) -> Array:
    return jnp.sqrt(sum(jnp.sum(arr * arr) for arr in jax.tree_util.tree_leaves(state)))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PDHGCarry:
    primal: OTState
    dual: OTState
    primal_bar: OTState
    phi_cache: Array
    iterations_used: Array
    converged: Array
    diagnostics: dict[str, Array]

    def tree_flatten(self):
        children = (
            self.primal,
            self.dual,
            self.primal_bar,
            self.phi_cache,
            self.iterations_used,
            self.converged,
            self.diagnostics,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def initialize_state(
    graph: GraphSpec,
    rho_a: Array,
    rho_b: Array,
    mean_ops: MeanOps,
    num_steps: int,
) -> OTState:
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
) -> tuple[OTState, dict[str, Array], Array, Array]:
    if initial_primal is None:
        init = initialize_state(graph, rho_a, rho_b, mean_ops, num_steps)
    else:
        init = initial_primal
    dual0 = _zero_state_like(init)
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
    )

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
        converged = should_check & (
            (diagnostics["primal_delta"] <= config.residual_tol)
            & (diagnostics["dual_delta"] <= config.residual_tol)
            & (diagnostics["max_constraint_residual"] <= config.feasibility_tol)
        )
        return PDHGCarry(
            primal=primal_next,
            dual=dual_next,
            primal_bar=primal_bar_next,
            phi_cache=phi_next,
            iterations_used=next_iter,
            converged=converged,
            diagnostics=diagnostics,
        )

    carry = lax.while_loop(cond_fn, body_fn, carry)
    return carry.primal, carry.diagnostics, carry.iterations_used, carry.converged
