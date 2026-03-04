# Getting Started

This guide is the fastest path to a correct first solve.

## Install

```bash
uv sync
```

Current assumptions:
- development is macOS-first,
- the package uses CPU-backed JAX,
- `jgot` enables JAX x64 mode on import,
- users do not need to set `JAX_ENABLE_X64=1` for normal runtime use.

## Smallest Complete Solve

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

## Endpoint Normalization

The solver does not take ordinary node masses directly. It expects densities
with respect to the stationary distribution `pi`.

Use this conversion:
- start from ordinary probability masses `mass`,
- convert to solver densities with `rho = mass / pi`.

The required normalization rule is:
- `sum(mass) == 1`,
- equivalently `sum(pi * rho) == 1`.

This is the most common source of input errors.

## How to Read the Result

The most useful output fields are:
- `sol.distance`: square root of the discrete action,
- `sol.converged`: whether the stopping tests were satisfied,
- `sol.iterations_used`: number of PDHG iterations actually performed,
- `sol.state.rho`: node densities over time,
- `sol.state.m`: edge fluxes over time.

If `sol.converged` is `False`, the returned state can still be useful for
inspection or debugging, but you should not treat the reported distance as a
fully trusted solved value.

## Where to Go Next

- For graph input semantics, see [Graph Model](graph-model.md).
- For the solver mechanics and scaling model, see [Solver Overview](solver-overview.md).
- For debugging failed or unstable runs, see [Debugging and Diagnostics](debugging-and-diagnostics.md).
