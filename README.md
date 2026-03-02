# JGOT (Jax Graph Optimal Transport)

Dynamic optimal transport on sparse reversible graphs in JAX. `jgot` currently
solves the two-endpoint dynamic optimal transport problem on finite reversible
graphs, following the time-discrete formulation of Erbar et al. (2020). JKO /
minimizing-movement wrappers are out of scope in the current implementation.

## What You Provide

`jgot` expects:

1. A finite sparse graph
2. A stationary distribution `pi`, or enough graph data for the library to
   infer it
3. Two endpoint densities `rho_a` and `rho_b`
4. A time discretization `num_steps`
5. A mean function, currently `LogMeanOps()`

Important normalization rule:

- Densities are represented with respect to `pi`
- They must satisfy `sum(pi * rho) == 1`

Graph assumptions:

- The graph must be connected
- Every directed edge with positive rate must have an explicit reverse edge
- The current runtime solver supports reversible graphs only

## Install

```bash
uv sync
```

Runtime assumptions:

- Current development target is macOS
- The package uses CPU-backed JAX

## Quickstart

This is the smallest complete solve.

```python
import jax.numpy as jnp

from jgot import (
    GraphSpec,
    LogMeanOps,
    OTConfig,
    OTProblem,
    TimeDiscretization,
    solve_ot,
)

graph = GraphSpec.from_undirected_weights(
    num_nodes=2,
    edge_u=[0],
    edge_v=[1],
    weight=[1.0],
)

mass_a = jnp.array([1.0, 0.0])
mass_b = jnp.array([0.0, 1.0])
rho_a = mass_a / graph.pi
rho_b = mass_b / graph.pi

problem = OTProblem(
    graph=graph,
    time=TimeDiscretization(num_steps=64),
    rho_a=rho_a,
    rho_b=rho_b,
    mean_ops=LogMeanOps(),
)

sol = solve_ot(problem, OTConfig())

print("distance:", float(sol.distance))
print("converged:", sol.converged)
print("iterations:", sol.iterations_used)
```

## Core API

### `GraphSpec`

Represents a sparse reversible graph stored internally as a directed edge list.
Most users should not instantiate `GraphSpec(...)` directly. Use one of the
constructors below.

#### `GraphSpec.from_undirected_weights(...)`

Inputs:

- `num_nodes`
- `edge_u`
- `edge_v`
- `weight`

Semantics:

- Each tuple `(edge_u[k], edge_v[k], weight[k])` defines an undirected
  conductance
- The constructor expands the graph into paired directed edges
- `pi` is derived from weighted degree normalization
- Reversibility is guaranteed by construction

Use this when your graph naturally comes from symmetric edge weights. This is
the preferred onboarding path.

#### `GraphSpec.from_directed_rates(...)`

Inputs:

- `num_nodes`
- `src`
- `dst`
- `q`
- optional `pi`
- `check_reversible`

Semantics:

- `src`, `dst`, and `q` define the directed rates `Q(x, y)`
- If `pi` is omitted, the library infers it under the reversibility assumption
- If reversibility fails, construction raises `ValueError`

Use this when you already have a directed reversible Markov rate graph.

#### `GraphSpec` fields

- `num_nodes`: number of graph nodes
- `num_edges`: number of directed edges
- `src`: source node for each directed edge
- `dst`: destination node for each directed edge
- `rev`: reverse-edge index for each directed edge
- `q`: directed edge rates
- `pi`: stationary distribution
- `out_rate`: row sum of outgoing rates at each node

### `TimeDiscretization`

`TimeDiscretization(num_steps: int)` defines the time grid for the dynamic OT
problem.

- `num_steps` must be at least `2`
- Smaller `num_steps` is cheaper but coarser
- Larger `num_steps` is more accurate but more expensive
- The time step is `h = 1 / num_steps`

### `LogMeanOps`

- `LogMeanOps()` is the only mean implementation currently provided
- It supplies the logarithmic mean used by the current solver
- Pass it as `OTProblem.mean_ops`

`MeanOps` is the abstract interface. Most users should use `LogMeanOps()`.

### `OTConfig`

`OTConfig` controls the solver and its internal subsolvers.

#### Solver step sizes

- `tau`
- `sigma`
- `relaxation`

Defaults are the correct starting point in most cases. The constraint
`tau * sigma < 1` must hold.

#### Convergence

- `max_iters`
- `check_every`
- `residual_tol`
- `feasibility_tol`

If `sol.converged` is `False`, increase `max_iters` first.

#### Nonlinear and linear subsolvers

- `newton_iters`
- `bisect_iters`
- `cg_max_iters`
- `cg_tol`
- `cg_warm_start`
- `cg_preconditioner`

These are advanced knobs. Most users should leave them at their defaults.

#### Warm start

- `warm_start`
- Allowed values:
  - `"linear_path"`
  - `"zero"`

`"linear_path"` is the practical default.

`tol` is kept only as a backward-compatible alias for `residual_tol`.

### `OTProblem`

`OTProblem` bundles the solve inputs:

- `graph`
- `time`
- `rho_a`
- `rho_b`
- `mean_ops`

`rho_a` and `rho_b` must have shape `(num_nodes,)` and satisfy
`sum(pi * rho) == 1`.

### `solve_ot(...)`

```python
solve_ot(problem: OTProblem, config: OTConfig = OTConfig()) -> OTSolution
```

This is the main entrypoint. It validates the endpoint densities, runs the
solver, and returns the estimated transport distance together with the full
time-discrete geodesic state.

### `OTSolution`

Key fields:

- `distance`: transport distance estimate
- `action`: discrete action before taking the square root
- `state`: full time-discrete transport path
- `iterations_used`: number of solver iterations used
- `converged`: convergence flag
- `diagnostics`: residuals and solver diagnostics

Most users will inspect:

- `state.rho`: node densities over time, shape `(N + 1, X)`
- `state.m`: edge fluxes over time, shape `(N, E)`

The remaining fields in `state` are advanced split variables:

- `vartheta`
- `rho_minus`
- `rho_plus`
- `rho_bar`
- `q_node`

## How to Use Your Own Data

### If your graph is undirected

This is the default pattern for user-supplied sparse graph data.

```python
import jax.numpy as jnp

from jgot import (
    GraphSpec,
    LogMeanOps,
    OTConfig,
    OTProblem,
    TimeDiscretization,
    solve_ot,
)

num_nodes = 5
edge_u = [0, 1, 2, 3]
edge_v = [1, 2, 3, 4]
weight = [1.0, 1.0, 1.0, 1.0]

graph = GraphSpec.from_undirected_weights(
    num_nodes=num_nodes,
    edge_u=edge_u,
    edge_v=edge_v,
    weight=weight,
)

# Ordinary probability masses in node coordinates.
mass_a = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
mass_b = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])

# Convert masses to densities with respect to pi.
rho_a = mass_a / graph.pi
rho_b = mass_b / graph.pi

problem = OTProblem(
    graph=graph,
    time=TimeDiscretization(num_steps=64),
    rho_a=rho_a,
    rho_b=rho_b,
    mean_ops=LogMeanOps(),
)

sol = solve_ot(problem, OTConfig())

mid_idx = sol.state.rho.shape[0] // 2
midpoint_density = sol.state.rho[mid_idx]

print("distance:", float(sol.distance))
print("midpoint density:", midpoint_density)
```

Checklist before solving:

1. `rho_a` and `rho_b` have length `num_nodes`
2. They are nonnegative
3. They are normalized against `graph.pi`

The normalization step is easy to get wrong:

- Start from ordinary node masses `mass`
- Convert to solver densities with `rho = mass / pi`
- Ensure `sum(mass) == 1`
- Equivalently, ensure `sum(pi * rho) == 1`

### If your graph is already a directed reversible rate matrix

```python
import jax.numpy as jnp

from jgot import (
    GraphSpec,
    LogMeanOps,
    OTConfig,
    OTProblem,
    TimeDiscretization,
    solve_ot,
)

src = [0, 1, 1, 2]
dst = [1, 0, 2, 1]
q = [2.0, 1.0, 1.0, 2.0]

graph = GraphSpec.from_directed_rates(
    num_nodes=3,
    src=src,
    dst=dst,
    q=q,
)

mass_a = jnp.array([1.0, 0.0, 0.0])
mass_b = jnp.array([0.0, 0.0, 1.0])
rho_a = mass_a / graph.pi
rho_b = mass_b / graph.pi

problem = OTProblem(
    graph=graph,
    time=TimeDiscretization(num_steps=64),
    rho_a=rho_a,
    rho_b=rho_b,
    mean_ops=LogMeanOps(),
)

sol = solve_ot(problem, OTConfig())

print("distance:", float(sol.distance))
print("converged:", sol.converged)
```

You can also pass a known `pi`:

```python
graph = GraphSpec.from_directed_rates(
    num_nodes=3,
    src=src,
    dst=dst,
    q=q,
    pi=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
)
```

Warnings:

- Every positive directed edge must have its reverse edge present
- Nonreversible inputs are rejected
- If your graph is not reversible, `jgot` is outside scope for that problem

## Data Preparation Tips

### From Python lists

Plain Python lists are accepted and converted internally.

### From NumPy arrays

`numpy.ndarray` is accepted. This is a common ingestion path when reading graph
data from files.

### From JAX arrays

`jax.numpy` arrays are also fine. Use `jnp.asarray(...)` if your pipeline is
already in JAX.

### From edge tables or CSVs

If your data starts as a table, extract the relevant columns into 1D arrays
before calling `GraphSpec`.

## How to Inspect the Result

```python
print("distance:", float(sol.distance))
print("converged:", sol.converged)
print("iterations:", sol.iterations_used)
print("diagnostics:", sol.diagnostics)

rho_path = sol.state.rho
m_path = sol.state.m

print("first density:", rho_path[0])
print("midpoint density:", rho_path[rho_path.shape[0] // 2])
print("first interval flow:", m_path[0])
```

Interpretation:

- `sol.state.rho[t]` is the node density at time slice `t`
- `sol.state.m[t]` is the edge flux on interval `t -> t + 1`

If `sol.converged` is `False`, the returned state is still useful for debugging
and inspection, but you should not treat `sol.distance` as fully trusted.

## Common Failure Modes

### `ValueError: graph must be connected`

The graph support is disconnected. Fix the edge set so every node lies in the
same connected component.

### `ValueError: missing reverse edge`

At least one directed edge is missing its reverse. Add the reverse directed
edge with a positive rate.

### `ValueError: graph is not reversible under the supplied stationary distribution`

Your `pi` and `q` do not satisfy detailed balance. Fix the rates, fix `pi`, or
use `GraphSpec.from_undirected_weights(...)` if the graph is naturally
undirected.

### `ValueError: rho_a must satisfy sum(pi * rho) == 1`

### `ValueError: rho_b must satisfy sum(pi * rho) == 1`

The endpoint density is not normalized with respect to `pi`. Convert node masses
to densities using `rho = mass / pi`.

### `sol.converged == False`

Increase `max_iters` first. Leave the advanced solver parameters at their
defaults unless you have a specific numerical reason to tune them.

## Examples

Runnable examples live in:

- [/Users/lyuwt/JaxGraphOT/examples/two_node_benchmark/run.py](/Users/lyuwt/JaxGraphOT/examples/two_node_benchmark/run.py)
- [/Users/lyuwt/JaxGraphOT/examples/cycle_neighbor_transport/run.py](/Users/lyuwt/JaxGraphOT/examples/cycle_neighbor_transport/run.py)
- [/Users/lyuwt/JaxGraphOT/examples/line_chain_transport/run.py](/Users/lyuwt/JaxGraphOT/examples/line_chain_transport/run.py)

See the examples guide for details:

- [/Users/lyuwt/JaxGraphOT/examples/README.md](/Users/lyuwt/JaxGraphOT/examples/README.md)

## Current Scope

The current implementation:

- uses JAX for the runtime solver
- runs the expensive solver iterations under JIT
- targets sparse reversible graphs

The current implementation does not provide:

- general nonreversible OT support
- GPU or Metal support
- a BCOO or BCSR sparse backend
- JKO / minimizing-movement support
- guarantees outside the currently tested scope
