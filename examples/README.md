# Examples

This folder contains runnable reproductions of the paper's small benchmark setups using the current solver implementation in this repository.

Important:
- These scripts reproduce the problem setups from the paper.
- They run the repository's current active solver in `jgot.solve_ot`.
- The current active solver is the sparse PDHG implementation in `pdhg.py`.
- The old linear-path construction is no longer the public solve path; it is only used as an internal warm-start.

## Scripts
- `two_node_benchmark/`: the 2-node logarithmic-mean benchmark with comparison to the closed-form reference integral.
- `cycle_neighbor_transport/`: neighboring-node transport on 3-cycle and 4-cycle graphs.
- `line_chain_transport/`: endpoint-to-endpoint transport on a 1D chain graph. This script still lightly regularizes the endpoint Dirac masses by default because the discrete problem is numerically stiff for exact Dirac-to-Dirac endpoints on coarse time grids.
- `directed_reversible_transport/`: a 3-node directed reversible example with asymmetric rates, demonstrating `GraphSpec.from_directed_rates(...)`.
- `large_grid_transport/`: a large 2D grid transport example (default 32x32, 1024 nodes) using localized corner blobs and full output plots.

## Run
- `uv run python examples/two_node_benchmark/run.py`
- `uv run python examples/cycle_neighbor_transport/run.py`
- `uv run python examples/line_chain_transport/run.py`
- `uv run python examples/directed_reversible_transport/run.py`
- `uv run python examples/large_grid_transport/run.py`

## Outputs
Each script saves:
- an `.npz` state dump under that example's local `output/` folder
- a node-mass `.png` plot
- an edge-flow `.png` plot
- a graph snapshot `.png` plot showing:
  - nodes and edges in the graph geometry
  - node area proportional to density
  - edge width proportional to flow magnitude
