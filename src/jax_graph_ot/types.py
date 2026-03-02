from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .means import MeanOps

Array = jax.Array


@dataclass(frozen=True)
class GraphSpec:
    num_nodes: int
    num_edges: int
    src: Array
    dst: Array
    rev: Array
    q: Array
    pi: Array
    out_rate: Array

    @classmethod
    def from_undirected_weights(
        cls,
        num_nodes: int,
        edge_u: Any,
        edge_v: Any,
        weight: Any,
        *,
        atol: float = 1e-12,
        rtol: float = 1e-10,
    ) -> "GraphSpec":
        from .graph import build_graph_from_undirected_weights

        return build_graph_from_undirected_weights(
            num_nodes,
            edge_u,
            edge_v,
            weight,
            atol=atol,
            rtol=rtol,
        )

    @classmethod
    def from_directed_rates(
        cls,
        num_nodes: int,
        src: Any,
        dst: Any,
        q: Any,
        *,
        pi: Any | None = None,
        check_reversible: bool = True,
        atol: float = 1e-12,
        rtol: float = 1e-10,
        tol_ratio: float = 1e-10,
    ) -> "GraphSpec":
        from .graph import build_graph_from_directed_rates

        return build_graph_from_directed_rates(
            num_nodes,
            src,
            dst,
            q,
            pi=pi,
            check_reversible=check_reversible,
            atol=atol,
            rtol=rtol,
            tol_ratio=tol_ratio,
        )


@dataclass(frozen=True)
class TimeDiscretization:
    num_steps: int

    def __post_init__(self) -> None:
        if self.num_steps < 2:
            raise ValueError("num_steps must be at least 2")

    @property
    def h(self) -> float:
        return 1.0 / float(self.num_steps)


@dataclass(frozen=True)
class OTConfig:
    tau: float = 0.95
    sigma: float = 0.95
    relaxation: float = 1.0
    warm_start: str = "linear_path"
    max_iters: int = 400
    check_every: int = 10
    tol: float | None = None
    residual_tol: float = 1e-8
    feasibility_tol: float = 1e-8
    newton_iters: int = 12
    bisect_iters: int = 20
    cg_max_iters: int = 200
    cg_tol: float = 1e-10
    cg_warm_start: bool = True
    cg_preconditioner: str = "jacobi"

    def __post_init__(self) -> None:
        if self.tau <= 0 or self.sigma <= 0:
            raise ValueError("tau and sigma must be positive")
        if self.tau * self.sigma >= 1.0:
            raise ValueError("tau * sigma must be strictly less than 1")
        if not 0.0 <= self.relaxation <= 1.0:
            raise ValueError("relaxation must lie in [0, 1]")
        if self.warm_start not in {"linear_path", "zero"}:
            raise ValueError("warm_start must be 'linear_path' or 'zero'")
        if self.max_iters <= 0 or self.check_every <= 0:
            raise ValueError("max_iters and check_every must be positive")
        if self.tol is not None:
            object.__setattr__(self, "residual_tol", self.tol)
        if self.residual_tol <= 0 or self.feasibility_tol <= 0:
            raise ValueError("residual_tol and feasibility_tol must be positive")
        if self.newton_iters <= 0 or self.bisect_iters <= 0:
            raise ValueError("newton_iters and bisect_iters must be positive")
        if self.cg_max_iters <= 0:
            raise ValueError("cg_max_iters must be positive")
        if self.cg_preconditioner != "jacobi":
            raise ValueError("cg_preconditioner must be 'jacobi' in v1")


@dataclass(frozen=True)
class OTProblem:
    graph: GraphSpec
    time: TimeDiscretization
    rho_a: Array
    rho_b: Array
    mean_ops: "MeanOps"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OTState:
    rho: Array
    m: Array
    vartheta: Array
    rho_minus: Array
    rho_plus: Array
    rho_bar: Array
    q_node: Array

    def tree_flatten(self):
        children = (
            self.rho,
            self.m,
            self.vartheta,
            self.rho_minus,
            self.rho_plus,
            self.rho_bar,
            self.q_node,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    def primal_norm(self) -> Array:
        terms = [
            jnp.sum(self.rho**2),
            jnp.sum(self.m**2),
            jnp.sum(self.vartheta**2),
            jnp.sum(self.rho_minus**2),
            jnp.sum(self.rho_plus**2),
            jnp.sum(self.rho_bar**2),
            jnp.sum(self.q_node**2),
        ]
        return jnp.sqrt(sum(terms))


@dataclass(frozen=True)
class OTSolution:
    distance: Array
    action: Array
    state: OTState
    iterations_used: int
    converged: bool
    diagnostics: dict[str, Array]
