"""Public data containers and configuration types for the dynamic OT solver."""

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
    """Sparse reversible graph stored as a directed edge list.

    The solver stores graphs as paired directed edges rather than as a dense
    matrix. Every directed edge must have an explicit reverse edge recorded by
    ``rev``. Most users should construct graphs through
    :meth:`from_undirected_weights` or :meth:`from_directed_rates` instead of
    instantiating this dataclass manually. Graphs built from symmetric weights
    are internally converted into a normalized reversible directed edge
    representation.

    Attributes:
        num_nodes: Number of graph nodes.
        num_edges: Number of directed edges.
        src: Source node for each directed edge, shape ``(E,)``.
        dst: Destination node for each directed edge, shape ``(E,)``.
        rev: Reverse-edge index for each directed edge, shape ``(E,)``.
        q: Directed edge rates, shape ``(E,)``.
        pi: Stationary distribution, shape ``(X,)``.
        out_rate: Outgoing rate sum at each node, shape ``(X,)``.
    """

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
        """Build a sparse reversible graph from undirected conductances.

        Each entry ``(edge_u[k], edge_v[k], weight[k])`` defines an undirected
        conductance. The constructor expands the graph into paired directed
        edges, computes ``pi`` from weighted degree normalization, and returns a
        graph that is reversible by construction.

        Args:
            num_nodes: Total number of nodes in the graph.
            edge_u: Source endpoint for each undirected edge.
            edge_v: Destination endpoint for each undirected edge.
            weight: Positive conductance assigned to each undirected edge.
            atol: Absolute tolerance used by the final invariant checks.
            rtol: Relative tolerance used by the final invariant checks.

        Returns:
            A validated :class:`GraphSpec` instance.

        Raises:
            ValueError: If the inputs are not one-dimensional, have mismatched
                lengths, contain invalid node ids, contain non-positive weights,
                contain self-loops, or do not define a connected graph.

        Notes:
            The symmetric input is interpreted as a conductance graph, not as a
            pre-built generator. The constructor computes the weighted degree
            ``d_x = sum_y w_xy`` and stores off-diagonal edge weights as
            ``q(x, y) = w_xy / d_x``. As a result, ``sum_{y != x} q(x, y) = 1``
            for every node, so the undirected constructor builds a
            unit-exit-rate random walk. If interpreted as a full generator, the
            implied diagonal would be ``-1`` at every node. This normalization
            is specific to :meth:`from_undirected_weights`; it is not a generic
            property of graphs built from directed rates.
        """

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
        """Build a sparse graph from directed rates ``Q(x, y)``.

        The inputs define a directed rate graph through ``src``, ``dst``, and
        ``q``. If ``pi`` is omitted, the constructor infers the stationary
        distribution under the reversibility assumption. Every positive edge must
        have an explicit reverse edge. When reversibility checks are enabled,
        nonreversible inputs raise ``ValueError``.

        Args:
            num_nodes: Total number of nodes in the graph.
            src: Source node for each directed edge.
            dst: Destination node for each directed edge.
            q: Positive directed edge rates.
            pi: Optional stationary distribution. If omitted, it is inferred
                from the reversible rates.
            check_reversible: Whether to validate detailed balance and
                stationarity before returning.
            atol: Absolute tolerance used by reversibility and stationarity
                checks.
            rtol: Relative tolerance used by reversibility and stationarity
                checks.
            tol_ratio: Tolerance for cycle-consistency checks during ``pi``
                inference.

        Returns:
            A :class:`GraphSpec` built from the directed rates.

        Raises:
            ValueError: If the arrays are malformed, contain invalid node ids,
                contain non-positive rates, omit a reverse edge, violate
                reversibility, or fail the stationary distribution checks.
        """

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
    """Time grid for the two-endpoint dynamic OT problem.

    ``num_steps`` is the number of time intervals. Larger values increase the
    temporal resolution and the computational cost of the solve.
    """

    num_steps: int

    def __post_init__(self) -> None:
        if self.num_steps < 2:
            raise ValueError("num_steps must be at least 2")

    @property
    def h(self) -> float:
        """Return the uniform time step size ``1 / num_steps``."""

        return 1.0 / float(self.num_steps)


@dataclass(frozen=True)
class OTConfig:
    """Solver configuration for the PDHG-based dynamic OT solve.

    Parameters are grouped by role:

    - PDHG step sizes: ``tau``, ``sigma``, ``relaxation``
    - Convergence checks: ``max_iters``, ``check_every``, ``residual_tol``,
      ``feasibility_tol``
    - Scalar root-solver controls: ``newton_iters``, ``bisect_iters``
    - Conjugate-gradient controls: ``cg_max_iters``, ``cg_tol``,
      ``cg_warm_start``, ``cg_preconditioner``
    - Warm start policy: ``warm_start``
    - Optional checkpoint tracing: ``record_debug_trace``

    Notes:
        ``tau * sigma`` must remain strictly less than ``1``. ``tol`` is kept
        only as a backward-compatible alias for ``residual_tol``. The defaults
        are the intended starting point for normal usage.
    """

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
    record_debug_trace: bool = False

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
    """Bundle the inputs for one dynamic OT solve.

    ``rho_a`` and ``rho_b`` are endpoint densities represented with respect to
    ``graph.pi``. They must each have shape ``(graph.num_nodes,)`` and satisfy
    ``sum(graph.pi * rho) == 1``.
    """

    graph: GraphSpec
    time: TimeDiscretization
    rho_a: Array
    rho_b: Array
    mean_ops: "MeanOps"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OTState:
    """Full split state used by the time-discrete solver.

    ``rho`` and ``m`` are the primary fields that most users inspect in the
    returned geodesic state. The remaining arrays are auxiliary split variables
    used by the PDHG implementation.

    Attributes:
        rho: Node densities over time, shape ``(N + 1, X)``.
        m: Edge fluxes over time, shape ``(N, E)``.
        vartheta: Mean-related edge variables, shape ``(N, E)``.
        rho_minus: Edge-local source densities, shape ``(N, E)``.
        rho_plus: Edge-local destination densities, shape ``(N, E)``.
        rho_bar: Time-averaged node densities, shape ``(N, X)``.
        q_node: Auxiliary node variables, shape ``(N, X)``.
    """

    rho: Array
    m: Array
    vartheta: Array
    rho_minus: Array
    rho_plus: Array
    rho_bar: Array
    q_node: Array

    def tree_flatten(self):
        """Return the JAX pytree children for this state."""

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
        """Reconstruct the state from JAX pytree children."""

        del aux_data
        return cls(*children)

    def primal_norm(self) -> Array:
        """Compute the Euclidean norm across all split state blocks."""

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
class OTDebugTrace:
    """Checkpointed diagnostic trace recorded during a PDHG solve.

    The arrays are fixed-size buffers allocated from ``max_iters`` and
    ``check_every``. Only the first ``num_records`` entries are valid.
    """

    iterations: Array
    action: Array
    continuity_residual: Array
    primal_delta: Array
    dual_delta: Array
    max_constraint_residual: Array
    ceh_cg_residual: Array
    ceh_cg_iters: Array
    min_vartheta: Array
    num_records: int


@dataclass(frozen=True)
class OTSolution:
    """Result returned by :func:`jgot.solve_ot`.

    ``distance`` is the square root of ``action``. ``state`` contains the
    time-discrete geodesic state, ``converged`` indicates whether the solver
    met its stopping criteria, and ``diagnostics`` stores residuals plus
    conjugate-gradient statistics from the inner projections. When enabled via
    :class:`OTConfig`, ``debug_trace`` stores checkpointed JIT-safe debug
    history for diagnosing convergence behavior.
    """

    distance: Array
    action: Array
    state: OTState
    iterations_used: int
    converged: bool
    diagnostics: dict[str, Array]
    debug_trace: OTDebugTrace | None = None
