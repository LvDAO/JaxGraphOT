# Documentation

`jgot` implements a time-discrete dynamic optimal transport solver on sparse
reversible graphs using JAX. The current scope is the two-endpoint dynamic OT
problem on finite reversible graphs. The implementation follows the split
formulation of Erbar et al. (2020), runs the expensive solver path under JIT,
and currently targets CPU-backed JAX.

Package link:
- [jgot on PyPI](https://pypi.org/project/jgot/)

## Start Here

- [Getting Started](getting-started.md): install, first solve, and endpoint normalization.
- [Graph Model](graph-model.md): what graph inputs mean, including symmetric weights vs directed rates.
- [Solver Overview](solver-overview.md): what algorithm is implemented and how it scales.
- [API Reference](api-reference.md): public types, functions, and debug-trace fields.
- [Examples Guide](examples-guide.md): what each shipped example demonstrates.
- [Debugging and Diagnostics](debugging-and-diagnostics.md): how to interpret residuals and debug traces.
- [Numerical Limitations](numerical-limitations.md): current limitations and large-grid caveats.
