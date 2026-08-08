"""Диагностика — главный продукт проекта, поэтому строится раньше правил обучения.

v0 (блок 0): регистратор метрик, нормы весов и активаций, разреженность.
Дальше сюда придут alignment, контроль runaway, гистограмма побед фильтров.
"""

from bioplast.diagnostics.metrics import MetricsRecorder
from bioplast.diagnostics.probes import (
    activation_stats,
    grad_norms,
    log_module_state,
    sparsity,
    weight_norms,
)

__all__ = [
    "MetricsRecorder",
    "activation_stats",
    "grad_norms",
    "log_module_state",
    "sparsity",
    "weight_norms",
]
