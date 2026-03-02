"""Averaging functions used by the transport action and ``K`` projection.

The current implementation provides the logarithmic mean only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array


class MeanOps(Protocol):
    """Abstract interface required by the solver's mean-dependent operations.

    Most users should use :class:`LogMeanOps`.
    """

    def theta(self, s: Array, t: Array) -> Array:
        """Evaluate the admissible mean."""

        ...

    def dtheta_ds(self, s: Array, t: Array) -> Array:
        """Differentiate the mean with respect to the first argument."""

        ...

    def dtheta_dt(self, s: Array, t: Array) -> Array:
        """Differentiate the mean with respect to the second argument."""

        ...

    def origin_supergrad_contains(self, z1: Array, z2: Array) -> Array:
        """Test the origin supergradient inclusion used by ``project_k``."""

        ...

    def project_k_top(self, p1: Array, p2: Array, p3: Array) -> tuple[Array, Array, Array]:
        """Project a point onto the upper boundary of the ``K`` set."""

        ...


@dataclass(frozen=True)
class LogMeanOps:
    """Logarithmic mean and its helper operations for the current solver.

    This implementation includes near-diagonal stabilization for the mean and
    its derivatives, plus scalar root solves used inside the pointwise ``K``
    projection.
    """

    eps_diag: float = 1e-6
    xi_max: float = 18.0
    newton_iters: int = 3
    bisect_iters: int = 24

    def theta(self, s: Array, t: Array) -> Array:
        """Evaluate the logarithmic mean ``theta(s, t)``.

        Args:
            s: First mean argument. Scalars or broadcast-compatible arrays.
            t: Second mean argument. Scalars or broadcast-compatible arrays.

        Returns:
            The logarithmic mean with numerically stabilized evaluation near the
            diagonal and zero output on the non-positive boundary.
        """

        s = jnp.asarray(s)
        t = jnp.asarray(t)
        positive = (s > 0) & (t > 0)
        close = positive & (jnp.abs(t - s) <= self.eps_diag * jnp.maximum(s, t))
        m = 0.5 * (s + t)
        u = (t - s) / jnp.maximum(s + t, jnp.finfo(m.dtype).tiny)
        series = m * (1.0 - (u * u) / 3.0)
        raw = (t - s) / (jnp.log(t) - jnp.log(s))
        out = jnp.where(close, series, raw)
        out = jnp.where(positive, out, 0.0)
        out = jnp.where((s == t) & (s >= 0), s, out)
        return out

    def dtheta_ds(self, s: Array, t: Array) -> Array:
        """Differentiate the logarithmic mean with respect to ``s``.

        Args:
            s: First mean argument. Scalars or broadcast-compatible arrays.
            t: Second mean argument. Scalars or broadcast-compatible arrays.

        Returns:
            The partial derivative ``d theta / d s`` evaluated with a
            near-diagonal series expansion when needed for numerical stability.
        """

        s = jnp.asarray(s)
        t = jnp.asarray(t)
        positive = (s > 0) & (t > 0)
        close = positive & (jnp.abs(t - s) <= self.eps_diag * jnp.maximum(s, t))
        m = 0.5 * (s + t)
        d = t - s
        series = (
            0.5
            + d / jnp.maximum(6.0 * m, jnp.finfo(m.dtype).tiny)
            + (d * d) / jnp.maximum(24.0 * m * m, jnp.finfo(m.dtype).tiny)
        )
        lr = jnp.log(t) - jnp.log(s)
        raw = ((t - s) / s - lr) / (lr * lr)
        out = jnp.where(close, series, raw)
        out = jnp.where(positive, out, 0.0)
        out = jnp.where((s == t) & (s > 0), 0.5, out)
        return out

    def dtheta_dt(self, s: Array, t: Array) -> Array:
        """Differentiate the logarithmic mean with respect to ``t``.

        Args:
            s: First mean argument. Scalars or broadcast-compatible arrays.
            t: Second mean argument. Scalars or broadcast-compatible arrays.

        Returns:
            The partial derivative ``d theta / d t`` evaluated with the same
            near-diagonal stabilization used for :meth:`dtheta_ds`.
        """

        s = jnp.asarray(s)
        t = jnp.asarray(t)
        positive = (s > 0) & (t > 0)
        close = positive & (jnp.abs(t - s) <= self.eps_diag * jnp.maximum(s, t))
        m = 0.5 * (s + t)
        d = t - s
        series = (
            0.5
            - d / jnp.maximum(6.0 * m, jnp.finfo(m.dtype).tiny)
            + (d * d) / jnp.maximum(24.0 * m * m, jnp.finfo(m.dtype).tiny)
        )
        lr = jnp.log(t) - jnp.log(s)
        raw = (lr - (t - s) / t) / (lr * lr)
        out = jnp.where(close, series, raw)
        out = jnp.where(positive, out, 0.0)
        out = jnp.where((s == t) & (s > 0), 0.5, out)
        return out

    def _beta(self, xi: Array) -> Array:
        """Evaluate the monotone scalar map inverted in the origin test."""

        s = jnp.exp(-0.5 * xi)
        t = jnp.exp(0.5 * xi)
        return self.dtheta_ds(s, t)

    def _invert_beta(self, target: Array) -> tuple[Array, Array]:
        """Invert ``_beta`` on a bounded interval using safeguarded root steps.

        Returns the approximate root in log-space together with a flag that
        indicates whether the target lay inside the search bracket.
        """

        lo = jnp.array(0.0, dtype=target.dtype)
        hi = jnp.array(self.xi_max, dtype=target.dtype)
        f_lo = self._beta(lo) - target
        f_hi = self._beta(hi) - target
        in_range = (f_lo <= 1e-12) & (f_hi >= -1e-12)

        def newton_body(_, state: tuple[Array, Array, Array]):
            left, right, xi = state
            value = self._beta(xi) - target
            deriv = jax.grad(self._beta)(xi)
            step = value / jnp.where(jnp.abs(deriv) > 1e-12, deriv, 1.0)
            cand = jnp.clip(xi - step, left, right)
            f_cand = self._beta(cand) - target
            move_left = f_cand < 0
            left = jnp.where(move_left, cand, left)
            right = jnp.where(move_left, right, cand)
            return left, right, cand

        left, right, _ = lax.fori_loop(
            0,
            self.newton_iters,
            newton_body,
            (lo, hi, 0.5 * (lo + hi)),
        )

        def bisect_body(_, state: tuple[Array, Array]):
            left, right = state
            mid = 0.5 * (left + right)
            f_mid = self._beta(mid) - target
            move_left = f_mid < 0
            left = jnp.where(move_left, mid, left)
            right = jnp.where(move_left, right, mid)
            return left, right

        left, right = lax.fori_loop(0, self.bisect_iters, bisect_body, (left, right))
        return 0.5 * (left + right), in_range

    def _top_surface_residual(self, p: Array, xi: Array) -> Array:
        """Return the scalar root residual for the ``K`` top-surface projection."""

        s = jnp.exp(0.5 * xi)
        t = jnp.exp(-0.5 * xi)
        w = jnp.asarray([s, t, self.theta(s, t)], dtype=p.dtype)
        n = jnp.asarray(
            [-self.dtheta_ds(s, t), -self.dtheta_dt(s, t), 1.0],
            dtype=p.dtype,
        )
        return jnp.dot(p, jnp.cross(w, n))

    def origin_supergrad_contains(self, z1: Array, z2: Array) -> Array:
        """Check the origin supergradient condition used by ``project_k``.

        Args:
            z1: First normalized slope component.
            z2: Second normalized slope component.

        Returns:
            A boolean-valued JAX array indicating whether the point lies in the
            origin supergradient of the logarithmic mean admissible set.
        """

        z1 = jnp.asarray(z1)
        z2 = jnp.asarray(z2)
        swap = z1 < z2
        primary = jnp.where(swap, z2, z1)
        secondary = jnp.where(swap, z1, z2)
        positive = (z1 > 0) & (z2 > 0)

        xi, in_range = self._invert_beta(primary)
        s = jnp.exp(-0.5 * xi)
        t = jnp.exp(0.5 * xi)
        needed = self.dtheta_dt(s, t)
        return positive & in_range & (secondary >= needed - 1e-12)

    def project_k_top(self, p1: Array, p2: Array, p3: Array) -> tuple[Array, Array, Array]:
        """Project a point onto the upper boundary of the ``K`` admissible set.

        Args:
            p1: First coordinate of the point to project.
            p2: Second coordinate of the point to project.
            p3: Third coordinate of the point to project.

        Returns:
            A tuple ``(q1, q2, q3)`` lying on the top surface defined by the
            logarithmic mean.
        """

        p = jnp.asarray([p1, p2, p3])
        lo0 = jnp.array(-self.xi_max, dtype=p.dtype)
        hi0 = jnp.array(self.xi_max, dtype=p.dtype)
        f_lo = self._top_surface_residual(p, lo0)
        f_hi = self._top_surface_residual(p, hi0)
        has_bracket = (
            (jnp.abs(f_lo) <= 1e-12) | (jnp.abs(f_hi) <= 1e-12) | (jnp.sign(f_lo) != jnp.sign(f_hi))
        )

        def newton_body(
            _,
            state: tuple[Array, Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array, Array]:
            left, right, f_left, f_right, xi = state
            value = self._top_surface_residual(p, xi)
            deriv = jax.grad(lambda arg: self._top_surface_residual(p, arg))(xi)
            step = value / jnp.where(jnp.abs(deriv) > 1e-12, deriv, 1.0)
            cand = jnp.clip(xi - step, left, right)
            f_cand = self._top_surface_residual(p, cand)
            same_side_as_left = jnp.sign(f_left) * jnp.sign(f_cand) > 0
            left = jnp.where(same_side_as_left, cand, left)
            f_left = jnp.where(same_side_as_left, f_cand, f_left)
            right = jnp.where(same_side_as_left, right, cand)
            f_right = jnp.where(same_side_as_left, f_right, f_cand)
            return left, right, f_left, f_right, cand

        left, right, f_left, f_right, _ = lax.fori_loop(
            0,
            self.newton_iters,
            newton_body,
            (lo0, hi0, f_lo, f_hi, jnp.array(0.0, dtype=p.dtype)),
        )

        def bisect_body(
            _,
            state: tuple[Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array]:
            left, right, f_left, f_right = state
            mid = 0.5 * (left + right)
            f_mid = self._top_surface_residual(p, mid)
            same_side_as_left = jnp.sign(f_left) * jnp.sign(f_mid) > 0
            left = jnp.where(same_side_as_left, mid, left)
            f_left = jnp.where(same_side_as_left, f_mid, f_left)
            right = jnp.where(same_side_as_left, right, mid)
            f_right = jnp.where(same_side_as_left, f_right, f_mid)
            return left, right, f_left, f_right

        left, right, _, _ = lax.fori_loop(
            0,
            self.bisect_iters,
            bisect_body,
            (left, right, f_left, f_right),
        )
        xi = 0.5 * (left + right)
        xi = jnp.where(
            has_bracket,
            xi,
            jnp.where(jnp.abs(f_lo) <= jnp.abs(f_hi), lo0, hi0),
        )
        s = jnp.exp(0.5 * xi)
        t = jnp.exp(-0.5 * xi)
        w = jnp.asarray([s, t, self.theta(s, t)], dtype=p.dtype)
        tau = jnp.maximum(jnp.dot(p, w) / jnp.maximum(jnp.dot(w, w), 1e-30), 0.0)
        proj = tau * w
        return proj[0], proj[1], proj[2]
