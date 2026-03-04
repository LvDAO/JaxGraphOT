from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_COMMON_PATH = Path(__file__).resolve().parents[1] / "examples" / "_common.py"
_COMMON_SPEC = importlib.util.spec_from_file_location("examples_common", _COMMON_PATH)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise RuntimeError("unable to load examples/_common.py for tests")
_COMMON = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON)

block_density = _COMMON.block_density
grid_graph = _COMMON.grid_graph
grid_layout = _COMMON.grid_layout
estimate_state_memory_bytes = _COMMON.estimate_state_memory_bytes
save_debug_trace_npz = _COMMON.save_debug_trace_npz
save_debug_trace_plot = _COMMON.save_debug_trace_plot


def test_grid_graph_counts() -> None:
    side = 4
    graph = grid_graph(side)
    assert graph.num_nodes == side * side
    assert graph.num_edges == 4 * side * (side - 1)


def test_grid_graph_is_row_normalized() -> None:
    graph = grid_graph(4)
    row_sum = np.zeros(graph.num_nodes, dtype=np.float64)
    np.add.at(row_sum, np.asarray(graph.src), np.asarray(graph.q))
    np.testing.assert_allclose(row_sum, np.ones(graph.num_nodes), atol=1e-12)


def test_block_density_normalizes_mass_correctly() -> None:
    graph = grid_graph(4)
    rho = block_density(graph, 4, rows=range(2), cols=range(2))
    mass = np.asarray(graph.pi) * rho
    assert np.all(rho >= 0)
    np.testing.assert_allclose(np.sum(mass), 1.0, atol=1e-12)


def test_grid_layout_shape() -> None:
    layout = grid_layout(4)
    assert layout.shape == (16, 2)
    assert float(np.min(layout)) >= 0.0
    assert float(np.max(layout)) <= 1.0


def test_memory_estimate_is_positive() -> None:
    graph_small = grid_graph(4)
    graph_large = grid_graph(5)
    bytes_small = estimate_state_memory_bytes(graph_small, 8)
    bytes_more_steps = estimate_state_memory_bytes(graph_small, 16)
    bytes_larger_graph = estimate_state_memory_bytes(graph_large, 8)
    assert isinstance(bytes_small, int)
    assert bytes_small > 0
    assert bytes_more_steps > bytes_small
    assert bytes_larger_graph > bytes_small


def test_save_debug_trace_helpers(tmp_path: Path) -> None:
    trace = SimpleNamespace(
        iterations=np.array([5, 10, 15], dtype=np.int32),
        action=np.array([1.0, np.inf, 3.0], dtype=np.float64),
        continuity_residual=np.array([1e-2, 1e-3, 1e-4], dtype=np.float64),
        primal_delta=np.array([1e-1, 1e-2, 1e-3], dtype=np.float64),
        dual_delta=np.array([1e-1, 1e-2, 1e-3], dtype=np.float64),
        max_constraint_residual=np.array([1e-2, 1e-3, 1e-4], dtype=np.float64),
        ceh_cg_residual=np.array([1e-3, 1e-4, 1e-5], dtype=np.float64),
        ceh_cg_iters=np.array([10, 12, 14], dtype=np.int32),
        min_vartheta=np.array([0.5, 0.1, -0.01], dtype=np.float64),
        num_records=3,
    )
    npz_path = save_debug_trace_npz(tmp_path, "trace_test", trace)
    png_path = save_debug_trace_plot(tmp_path, "trace_test", trace, title="trace test")
    assert npz_path.exists()
    assert png_path.exists()
