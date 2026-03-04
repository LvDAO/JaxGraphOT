# Numerical Limitations

This project is usable now, but it is important to understand its current
limits.

## Current Scope and Limits

The current implementation supports:
- finite sparse reversible graphs,
- CPU-backed JAX,
- the two-endpoint dynamic OT problem.

It does not currently provide:
- nonreversible transport,
- GPU guarantees,
- entropy-regularized OT,
- static coupling OT,
- JKO / minimizing movement.

## Large-Grid Stability Caveats

The solver can run on larger grids in principle, but the current implementation
is not yet robust enough to treat `8x8` or `32x32` examples as production-stable
defaults.

Important warning:
- small continuity residual does **not** imply a finite action.

The raw PDHG iterate can become singular even when continuity is already nearly
satisfied.

## Paper Fidelity vs Practical Approximations

The solver follows the paper's split formulation closely, but the current code
still makes practical numerical simplifications.

In particular:
- pointwise subproblems are implemented with fixed-iteration scalar solves,
- the `CE_h` projection is implemented in a numerically practical JAX form,
  not as a literal weighted-operator transcription of the paper.

These choices are reasonable for a first implementation, but they can matter on
larger or stiffer problems.

## Typical Singular Failure Mode

The current observed large-grid failure mode is:
- continuity becomes small,
- inner CG becomes accurate,
- `min_vartheta` shrinks toward `0`,
- action becomes non-finite.

This indicates:
- a degenerating raw outer iterate,
- not necessarily a failure of the continuity projection.

## What to Do When a Run Becomes Singular

The practical response is:
- reduce `steps`,
- increase `blob_size`,
- increase `cg_max_iters`,
- use `record_debug_trace=True`,
- inspect `min_vartheta` in the trace.

If `min_vartheta` keeps collapsing, the run is heading toward a singular state.

## Recommended Practical Stance

Treat large-grid runs as:
- diagnostic,
- exploratory,
- useful for observing solver behavior,

but not yet as guaranteed stable benchmark-quality solves under default settings.
