# Solver Overview

## What Is Implemented

`jgot` implements a time-discrete dynamic optimal transport solver on sparse
reversible graphs, following the split convex formulation from Erbar et al.
(2020).

The current algorithm has these main pieces:
- a time-discrete graph Wasserstein formulation,
- a split state containing primal and auxiliary variables,
- a PDHG (Chambolle-Pock style) outer loop,
- a `CE_h` continuity projection solved by conjugate gradient,
- pointwise projections for the remaining split constraints.

## What It Is Not

The current solver is not:
- Sinkhorn,
- entropy regularized OT,
- a static coupling solver,
- a JKO / minimizing-movement implementation.

It solves a constrained dynamic OT problem on graphs, not an entropic transport
problem.

## JIT Behavior

The expensive numerical path runs under JAX JIT.

In particular:
- the PDHG iteration runs inside JAX control flow,
- the expensive per-iteration linear algebra and projections are JAX-native,
- light orchestration and result packaging stay outside the hottest numerical path,
- debug tracing is recorded inside the JIT path using fixed-size buffers.

So the heavy work remains compiled, even when debug tracing is enabled.

## Current Split Structure

The solver works on a split state with these main blocks:
- `rho`: node densities over time,
- `m`: edge fluxes,
- `vartheta`: mean-related edge variables,
- `rho_minus`, `rho_plus`, `rho_bar`, `q_node`: auxiliary split variables.

The user usually only needs:
- `rho`,
- `m`,
- diagnostics,
- and optionally the debug trace.

## Cost and Scaling

The solver is sparse in graph structure, but it is not cheap on large time grids.

Rough scaling:
- graph storage is small,
- state storage scales like `O(NX + NE)`,
- total persistent solver storage is dominated by several copies of the split state,
- large runs become expensive mainly through `N * E`.

Where:
- `X` = number of nodes,
- `E` = number of directed edges,
- `N` = number of time steps.

This means:
- increasing graph size matters,
- but increasing the time grid often hurts just as much or more.

## Practical Consequence

The current implementation is comfortable on:
- toy graphs,
- small cycles,
- small path graphs,
- moderate debug examples.

It can run on larger grids, but larger grids are best treated as:
- experimental / diagnostic cases,
- not as guaranteed production-scale defaults.
