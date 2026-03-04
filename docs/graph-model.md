# Graph Model

`jgot` supports sparse, finite, reversible graphs. The solver stores graphs as
paired directed edges internally, but the user can construct them either from
symmetric weights or from directed rates.

## Graph Assumptions

All supported runtime graphs must be:
- finite,
- sparse,
- connected,
- reversible,
- explicitly paired so every positive directed edge has a reverse edge.

The stationary distribution is stored as a 1D array `pi`, and endpoint
densities are represented with respect to `pi`.

## `GraphSpec.from_undirected_weights(...)`

This constructor treats the input as a **symmetric conductance / weight graph**.
The input is not treated as a pre-built Markov generator.

Given symmetric weights `w_xy`, the constructor:
1. computes the weighted degree `d_x = sum_y w_xy`,
2. sets the stationary distribution `pi(x)` proportional to `d_x`,
3. stores off-diagonal directed weights as `q(x, y) = w_xy / d_x`.

As a result:
- the off-diagonal row sum is `sum_{y != x} q(x, y) = 1`,
- the undirected constructor builds a **unit-exit-rate random walk**,
- if interpreted as a full generator, the implied diagonal would be `-1`.

This normalization is a modeling choice for the symmetric-input path. It is why
symmetric graph weights become a reversible directed edge model internally.

Use this path when you naturally have:
- a symmetric affinity graph,
- a conductance graph,
- a symmetric sparse kernel graph.

## `GraphSpec.from_directed_rates(...)`

This constructor treats the input as **off-diagonal directed rates**.

That means:
- the input is not interpreted as a row-stochastic transition matrix,
- it is interpreted as the off-diagonal part of a continuous-time jump process,
- the full generator is implicit, with diagonal equal to minus the outgoing row sum.

If `pi` is omitted:
- `jgot` infers it under the reversibility assumption using sparse log-ratio
  propagation.

If reversibility fails:
- graph construction raises `ValueError`.

Use this path when you already have a reversible directed rate graph.

## Relation to Diffusion Maps

If your pipeline comes from diffusion maps, the safe rule is:
- pass the **last symmetric positive graph before row-normalization**.

That usually means:
- pass the symmetric kernel / affinity matrix, or
- pass the symmetric alpha-normalized kernel,
- but do **not** pass the row-normalized diffusion kernel as if it were already
  a generator unless you intentionally want to reinterpret it as rates.

Recommended practice:
- if you still have the symmetric sparse graph, use `from_undirected_weights(...)`.
- if you only have a reversible directed rate model, use `from_directed_rates(...)`.

## Stationary Distribution Convention

In code, `pi` is stored as a 1D array.

Mathematically, the convention is:
- `pi` is the **left stationary distribution**,
- the intended action is `pi @ Q`, not `Q @ pi`.

So the correct mental model is:
- `pi` acts as a row vector in the stationary equations,
- while `rho` is the density with respect to `pi`.

## Density Convention

Endpoint densities `rho_a` and `rho_b` are not ordinary masses.
They are densities with respect to `pi`, and they must satisfy:
- `sum(pi * rho) == 1`.

To convert ordinary node masses `mass` into the solver convention:
- `rho = mass / pi`.

## Internal Representation

Internally, `GraphSpec` stores:
- `src`,
- `dst`,
- `rev`,
- `q`,
- `pi`,
- `out_rate`.

This is a directed edge-list representation, not a dense matrix and not a
CSR/COO wrapper class. The solver operates on these arrays directly.
