# JGOT (JAX Graph Optimal Transport)

[![PyPI](https://img.shields.io/pypi/v/jgot.svg)](https://pypi.org/project/jgot/)

`jgot` solves two-endpoint dynamic optimal transport on sparse reversible graphs
using JAX, following the time-discrete formulation of
[Erbar et al. (2020)](https://arxiv.org/abs/1707.06859).

## Install

PyPI package:
- [jgot on PyPI](https://pypi.org/project/jgot/)

Core library:
```bash
pip install jgot
```

Examples (plotting dependencies included):
```bash
pip install "jgot[examples]"
```

Development environment:
```bash
uv sync --group dev
```

## Minimal Example

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
print(float(sol.distance), sol.converged, sol.iterations_used)
```

Important:
- Densities are represented with respect to `pi`.
- Endpoints must satisfy `sum(pi * rho) == 1`.

## Documentation

Detailed docs live under [docs](docs/index.md).

Recommended starting points:
- [Getting Started](docs/getting-started.md)
- [Graph Model](docs/graph-model.md)
- [API Reference](docs/api-reference.md)
- [Examples Guide](docs/examples-guide.md)
- [Debugging and Diagnostics](docs/debugging-and-diagnostics.md)
- [Numerical Limitations](docs/numerical-limitations.md)

## Examples

Runnable scripts:
- `examples/two_node_benchmark/run.py`
- `examples/cycle_neighbor_transport/run.py`
- `examples/line_chain_transport/run.py`
- `examples/directed_reversible_transport/run.py`
- `examples/large_grid_transport/run.py`

See [examples/README.md](examples/README.md) for commands and outputs.
