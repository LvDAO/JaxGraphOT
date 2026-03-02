# AGENTS.md

## Purpose
This repository implements the dynamic optimal transport solver from Erbar et al. (2020) on sparse reversible graphs using JAX.

## Scope
- First delivery scope is the core two-endpoint dynamic OT solver (distance + geodesic).
- JKO / minimizing-movement wrappers are out of scope for the first implementation unless explicitly requested later.

## Platform
- Develop on macOS only for now.
- Use `uv` for environment and dependency management.
- Use CPU-backed JAX first; do not depend on `jax-metal` in the initial implementation.
- Run all development and test commands with `JAX_ENABLE_X64=1`.

## Graph Model
- Supported runtime graphs are finite, sparse, and reversible.
- Every directed edge with positive weight must have an explicit reverse edge.
- Densities are represented with respect to the stationary distribution `pi` and must satisfy `sum(pi * rho) == 1`.

## Graph Construction Rules
- Prefer constructing graphs from undirected conductances when possible:
  - `pi(x)` is the weighted degree normalized to sum to 1.
  - `Q(x,y) = w_xy / pi(x)`.
- If directed rates are provided without `pi`, infer `pi` by sparse log-ratio propagation over a BFS spanning tree:
  - set one root log-mass to 0,
  - propagate `log_pi[y] = log_pi[x] + log(Qxy) - log(Qyx)`,
  - check cycle consistency on non-tree edges,
  - normalize after exponentiating.
- If directed rates are provided with `pi`, validate reversibility directly by checking detailed balance edgewise.
- Reject nonreversible graphs in v1.

## Numerical Rules
- Keep the runtime fully JAX-compatible and end-to-end JIT-safe.
- Do not route core solves through SciPy / NumPy solvers in the runtime path.
- Preserve sparse-graph structure as edge lists; do not materialize dense `X x X` transport variables.
- Use fixed-iteration scalar root solvers and JAX control flow (`lax.scan`, `lax.cond`, `lax.fori_loop`) inside compiled kernels.
- Keep all shapes static for a compiled solve: node count, edge count, and time-step count must not change mid-run.

## Solver Architecture
- Match the paper's split formulation and proximal subproblems closely.
- Implement the logarithmic mean first; add other averaging functions only behind the same interface.
- Use:
  - Thomas algorithm for the `Javg` time-only tridiagonal systems,
  - matrix-free conjugate gradient for the gauge-fixed `CE_h` elliptic projection,
  - pointwise scalar root solves for `A*` and `K` projections.

## Testing Expectations
- Add unit tests for every projection / proximal submodule.
- Add explicit tests for:
  - reverse-edge pairing,
  - connectivity,
  - stationary distribution inference,
  - detailed-balance validation,
  - nonreversible graph rejection.
- Add end-to-end tests for distance symmetry, mass preservation, endpoint satisfaction, and the 2-node benchmark.
- Every new numerical kernel should be tested both without JIT and with `jax.jit`.

## Standard Commands
- `JAX_ENABLE_X64=1 uv sync`
- `JAX_ENABLE_X64=1 uv run pytest -q`
- `JAX_ENABLE_X64=1 uv run ruff check .`

## Change Discipline
- Keep public dataclass fields and solver return shapes stable.
- When changing numerics, update the corresponding benchmark tolerances and explain why.
- Do not add JKO-specific abstractions until the core OT solver is stable and tested.
