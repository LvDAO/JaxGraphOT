# Debugging and Diagnostics

This file explains how to interpret solver diagnostics and how to use the debug
trace for unstable or under-converged runs.

## Basic Diagnostics

`OTSolution.diagnostics` reports the latest checkpoint values.

Key fields:
- `continuity_residual`: how well the discrete continuity equation is satisfied
- `primal_delta`: change in the primal iterate
- `dual_delta`: change in the dual iterate
- `max_constraint_residual`: worst current feasibility residual
- `ceh_cg_residual`: residual of the inner `CE_h` conjugate-gradient solve
- `ceh_cg_iters`: inner CG iterations used at the latest checkpoint

Mode note:
- `primal_delta` and `dual_delta` use weighted paper-style norms.
- `OTConfig.numerics_mode` is compatibility-only and must be `"paper"`.

## What `converged=False` Means

A run returns `converged=False` when the stopping rules were not all satisfied.
That usually means one of these:
- the solve hit `max_iters`,
- the inner `CE_h` solve is underpowered,
- the outer PDHG iterate is still moving,
- the current iterate is numerically singular.

Important:
- a small continuity residual alone does not imply a solved run.

## Enabling the Debug Trace

Use:

```python
from jgot import OTConfig, solve_ot

sol = solve_ot(problem, OTConfig(record_debug_trace=True))
trace = sol.debug_trace
```

The trace is recorded:
- at checkpoint iterations only,
- inside the JIT-compiled solve path,
- in fixed-size arrays.

Only the first `trace.num_records` entries are valid.

## How to Read the Trace

Most useful fields:
- `trace.iterations`
- `trace.action`
- `trace.continuity_residual`
- `trace.primal_delta`
- `trace.dual_delta`
- `trace.min_vartheta`

### Rising action
A rising action does **not** mean a bug by itself.
The solver is PDHG, not a monotone descent method on the action.
The action is evaluated on the current raw iterate, and raw objective values are
not guaranteed to decrease.

### Small continuity residual but non-finite action
This is the most important failure signature in the current implementation.
If:
- `continuity_residual` is already small,
- but `action` becomes `inf`,
- and `min_vartheta` approaches `0`,

then the issue is usually:
- outer-iterate degeneracy on the `K` / action side,
- not a failure of the continuity projection.

### Shrinking `min_vartheta`
This is the best early warning for a singular run.
If `min_vartheta` steadily approaches `0`, the iterate is moving toward a state
where the action denominator collapses.

## The `8x8` Large-Grid Failure Pattern

The current large-grid diagnostics showed a representative `8x8` failure mode:
- continuity residual became very small,
- inner CG became very accurate,
- but the action still became non-finite,
- because `min_vartheta` collapsed toward zero.

Interpretation:
- the continuity side was already working,
- the raw outer iterate was still drifting,
- the instability was on the `K` / action side.

This is why the trace records both:
- continuity,
- and `min_vartheta`.

## Practical Tuning Order

When a run is unstable or too slow, tune in this order:

1. `cg_max_iters`
2. `max_iters`
3. `steps`
4. `blob_size`
5. only then, if needed:
   - `tau`
   - `sigma`
   - `relaxation`

Reason:
- first ensure the inner continuity projection is not the bottleneck,
- then reduce outer stiffness,
- only then touch the PDHG step sizes.

## Large-Grid Example Debugging

The large-grid example supports:

```bash
uv run python examples/large_grid_transport/run.py --debug-trace
```

This writes:
- a trace `.npz`
- a trace `.png`

Use that path when:
- `converged=False`,
- action looks suspicious,
- or you want to see whether the run is failing on the continuity side or the
  `K` / action side.
