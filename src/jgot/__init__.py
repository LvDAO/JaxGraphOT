"""Public API surface for ``jgot``.

Import the user-facing types and ``solve_ot`` from this module rather than from
internal implementation modules.
"""

from . import _jax_config as _jax_config
from .means import LogMeanOps, MeanOps
from .solver import solve_ot
from .types import (
    GraphSpec,
    OTConfig,
    OTDebugTrace,
    OTProblem,
    OTSolution,
    OTState,
    TimeDiscretization,
)

__all__ = [
    "GraphSpec",
    "LogMeanOps",
    "MeanOps",
    "OTConfig",
    "OTDebugTrace",
    "OTProblem",
    "OTSolution",
    "OTState",
    "TimeDiscretization",
    "solve_ot",
]
