"""Assembly Theory Metrics Module.

Provides data models and tracking for Assembly Theory metrics,
which quantify the complexity and selection history of assembled objects.

Components:
- assembly: Core data models for Assembly Theory
- assembly_tracker: Decision history tracking and path analysis

Re-exports from titan.prometheus_metrics:
- get_metrics: Get the global Prometheus MetricsCollector
- MetricsCollector: Centralized metrics collector class
"""

from typing import Any, Protocol, cast

from titan.metrics.assembly import (
    AssemblyMetrics,
    AssemblyPath,
    AssemblyStep,
    SelectionSignal,
)
from titan.metrics.assembly_tracker import AssemblyTracker


class _PrometheusMetricsModule(Protocol):
    MEMORY_MGET_TOTAL: Any
    EMBEDDING_WAIT_TIME: Any

    def get_metrics(self) -> Any: ...
    def get_metrics_text(self) -> str: ...
    def start_metrics_server(self, port: int = 9100, host: str = "0.0.0.0") -> None: ...


# Re-export Prometheus metrics from the top-level module
# This handles the shadowing issue where titan/metrics/ package
# shadows titan/metrics.py module
def _get_prometheus_metrics() -> _PrometheusMetricsModule:
    """Lazy import to avoid circular dependency at module load time."""
    import importlib.util
    import sys

    # Manually load titan/metrics.py since it's shadowed by this package
    spec = importlib.util.spec_from_file_location(
        "titan.prometheus_metrics",
        __file__.replace("__init__.py", "../metrics.py").replace("metrics/../", ""),
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["titan.prometheus_metrics"] = module
        spec.loader.exec_module(module)
        return cast(_PrometheusMetricsModule, module)
    raise ImportError("Could not load titan/metrics.py")


# Lazy loading for get_metrics to avoid circular imports
_prometheus_module: _PrometheusMetricsModule | None = None


def get_metrics() -> Any:
    """Get the global Prometheus MetricsCollector."""
    global _prometheus_module
    if _prometheus_module is None:
        _prometheus_module = _get_prometheus_metrics()
    return _prometheus_module.get_metrics()


def get_metrics_text() -> str:
    """Get metrics in Prometheus text format."""
    global _prometheus_module
    if _prometheus_module is None:
        _prometheus_module = _get_prometheus_metrics()
    return _prometheus_module.get_metrics_text()


def start_metrics_server(port: int = 9100, host: str = "0.0.0.0") -> None:
    """Start Prometheus metrics HTTP server."""
    global _prometheus_module
    if _prometheus_module is None:
        _prometheus_module = _get_prometheus_metrics()
    _prometheus_module.start_metrics_server(port, host)


# Direct exports for specific metrics
MEMORY_MGET_TOTAL: Any | None = None
EMBEDDING_WAIT_TIME: Any | None = None


def _init_metrics() -> None:
    global MEMORY_MGET_TOTAL, EMBEDDING_WAIT_TIME, _prometheus_module
    if _prometheus_module is None:
        _prometheus_module = _get_prometheus_metrics()
    if hasattr(_prometheus_module, "MEMORY_MGET_TOTAL"):
        MEMORY_MGET_TOTAL = _prometheus_module.MEMORY_MGET_TOTAL
    if hasattr(_prometheus_module, "EMBEDDING_WAIT_TIME"):
        EMBEDDING_WAIT_TIME = _prometheus_module.EMBEDDING_WAIT_TIME


# Initialize immediately
try:
    _init_metrics()
except Exception:
    pass

__all__ = [
    "AssemblyMetrics",
    "AssemblyPath",
    "AssemblyStep",
    "AssemblyTracker",
    "SelectionSignal",
    "get_metrics",
    "get_metrics_text",
    "start_metrics_server",
    "MEMORY_MGET_TOTAL",
    "EMBEDDING_WAIT_TIME",
]
