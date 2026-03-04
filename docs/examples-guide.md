# Examples Guide

The repository ships runnable examples under `/examples`. Each example writes
artifacts to its own local `output/` directory.

## Examples Included

## Two-Node Benchmark
Path:
- `examples/two_node_benchmark/run.py`

Purpose:
- reproduces the 2-node logarithmic-mean benchmark,
- compares against the known closed-form reference integral.

Look at:
- reported distance,
- reference agreement,
- node-mass and edge-flow plots.

## Cycle Neighbor Transport
Path:
- `examples/cycle_neighbor_transport/run.py`

Purpose:
- compares neighbor transport on 3-cycles and 4-cycles,
- illustrates qualitative path-selection behavior on small graphs.

Look at:
- midpoint mass distribution,
- graph snapshots,
- whether mass visits the longer route.

## Line Chain Transport
Path:
- `examples/line_chain_transport/run.py`

Purpose:
- endpoint-to-endpoint transport on a 1D chain.

Notes:
- the example lightly regularizes exact Dirac endpoints because coarse time
  grids make exact Dirac-to-Dirac runs numerically stiff.

## Directed Reversible Transport
Path:
- `examples/directed_reversible_transport/run.py`

Purpose:
- demonstrates a non-symmetric but reversible directed graph,
- shows `GraphSpec.from_directed_rates(...)` in a supported case.

Look at:
- inferred `pi`,
- asymmetric directed rates,
- the resulting node and flow outputs.

## Large Grid Transport
Path:
- `examples/large_grid_transport/run.py`

Purpose:
- stress-test style example on a square grid,
- default graph is `32x32` (1024 nodes),
- uses localized corner blobs instead of exact corner Diracs.

Useful options:
- `--side`
- `--steps`
- `--blob-size`
- `--max-iters`
- `--check-every`
- `--cg-max-iters`
- `--debug-trace`

### Standard outputs
The script writes:
- state dump `.npz`
- node heatmap `.png`
- edge-flow heatmap `.png`
- graph snapshot `.png`

### Additional debug outputs
When `--debug-trace` is enabled, it also writes:
- debug trace `.npz`
- debug trace `.png`

These show checkpointed:
- action,
- continuity residual,
- and other diagnostic arrays saved in the trace file.

## Which Example to Use First

Recommended order:
1. two-node benchmark
2. cycle transport
3. directed reversible transport
4. line chain transport
5. large-grid transport

This order goes from easiest to most numerically demanding.

## Related Docs

- For the short example index, see `examples/README.md`.
- For trace interpretation, see [Debugging and Diagnostics](debugging-and-diagnostics.md).
- For current stability caveats on larger graphs, see [Numerical Limitations](numerical-limitations.md).
