# API Reference

This file summarizes the current public API surface of `jgot`.

## `GraphSpec`

`GraphSpec` stores a sparse reversible graph as a directed edge list.

Important fields:
- `num_nodes`
- `num_edges`
- `src`
- `dst`
- `rev`
- `q`
- `pi`
- `out_rate`

Recommended construction paths:
- `GraphSpec.from_undirected_weights(...)`
- `GraphSpec.from_directed_rates(...)`

## `TimeDiscretization`

`TimeDiscretization(num_steps: int)` defines the number of time intervals.

Practical meaning:
- larger `num_steps` gives a finer path,
- larger `num_steps` also makes the problem harder and more memory-intensive.

The step size is:
- `h = 1 / num_steps`

## `LogMeanOps`

`LogMeanOps()` is the only public mean implementation in the current project.
It provides the logarithmic mean and the helper operations needed by the `K`
projection.

Most users should use `LogMeanOps()` directly.

## `OTConfig`

`OTConfig` controls the PDHG solver and its inner subsolvers.

### PDHG step sizes
- `tau`: primal step size
- `sigma`: dual step size
- `relaxation`: over-relaxation factor

These are the closest thing this solver has to “learning-rate-like” parameters.
They control how aggressively the outer primal-dual method moves.

Constraint:
- `tau * sigma < 1`

### Iteration control
- `max_iters`
- `check_every`

These control how long the outer solve runs and how often convergence is checked.

### Inner CG controls
- `cg_max_iters`
- `cg_tol`
- `cg_warm_start`
- `cg_preconditioner`

These control the inner `CE_h` projection solve.
- `cg_preconditioner` accepts:
  - `"jacobi"` (default)
  - `"block_jacobi"` (time-block tridiagonal preconditioner)

### Numerical mode
- `numerics_mode`
  - `"paper"` (default and only supported value)

`numerics_mode` is kept for compatibility.
Passing `"legacy"` now raises:
- `legacy mode has been removed; use numerics_mode='paper'`

### Root-solver controls
- `newton_iters`
- `bisect_iters`

These affect the fixed-iteration scalar solves used inside pointwise projections.

### Warm start
- `warm_start`
  - `"linear_path"`
  - `"zero"`

### Debug tracing
- `record_debug_trace`

When enabled, the solver records checkpointed debug history inside the JIT path
and returns it through `OTSolution.debug_trace`.

## `OTProblem`

`OTProblem` bundles:
- `graph`
- `time`
- `rho_a`
- `rho_b`
- `mean_ops`

`rho_a` and `rho_b` must:
- have shape `(num_nodes,)`,
- be nonnegative,
- satisfy `sum(pi * rho) == 1`.

## `solve_ot(...)`

```python
solve_ot(problem: OTProblem, config: OTConfig = OTConfig()) -> OTSolution
```

This is the main public entrypoint.

It:
- validates endpoint densities,
- constructs a warm start,
- runs the PDHG-based solver path,
- returns the final state, diagnostics, and optional trace.

## `OTSolution`

Important fields:
- `distance`
- `action`
- `state`
- `iterations_used`
- `converged`
- `diagnostics`
- `debug_trace`

### `distance`
The square root of the discrete action.

### `action`
The current discrete action evaluated on the returned state.
This can become non-finite if the iterate is singular.

### `state`
The returned time-discrete transport path.
Most users inspect:
- `state.rho`
- `state.m`

### `converged`
Whether the stopping conditions were met.
A finite-looking state can still have `converged=False`.

### `diagnostics`
A dictionary containing the latest checkpoint values for:
- `primal_delta`
- `dual_delta`
- `continuity_residual`
- `k_violation`
- `endpoint_residual`
- `max_constraint_residual`
- `ceh_cg_residual`
- `ceh_cg_iters`

Note:
- `primal_delta` and `dual_delta` use the weighted paper-style Hilbert-space
  scaling.

### `debug_trace`
Optional checkpoint history returned when `record_debug_trace=True`.

## `OTDebugTrace`

`OTDebugTrace` is a fixed-size checkpoint buffer.

Important rule:
- only the first `num_records` entries are valid.

Recorded fields:
- `iterations`
- `action`
- `continuity_residual`
- `primal_delta`
- `dual_delta`
- `max_constraint_residual`
- `ceh_cg_residual`
- `ceh_cg_iters`
- `min_vartheta`
- `num_records`

This is the main tool for understanding why a run did not converge or why the
returned action became non-finite.

## Related Docs

- For graph semantics, see [Graph Model](graph-model.md).
- For solver behavior, see [Solver Overview](solver-overview.md).
- For debugging, see [Debugging and Diagnostics](debugging-and-diagnostics.md).
- For runnable scripts, see [Examples Guide](examples-guide.md).
