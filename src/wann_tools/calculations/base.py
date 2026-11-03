from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from wann_tools.config import AppConfig
from wann_tools.context import CalculationContext
from wann_tools.executors.base import Executor

@dataclass
class ResourceEstimate:
    # Memory required by one worker, not the entire job.
    peak_memory_bytes_per_worker: int

    # Total number of independent work items.
    total_work_items: int

    # Optional calibrated wall-clock estimate.
    estimated_seconds: float | None = None
    # Human-readable assumptions and warnings.
    assumptions: list[str] | None = None

    fixed_memory_bytes_per_worker: int = 0
    working_memory_bytes_per_worker: int = 0
    overhead_margin: float = 0.20


class Calculation(ABC):
    """Interface implemented by each calculation type."""

    # Unique name used in the YAML configuration.
    name: str

    @abstractmethod
    def validate(self, config: AppConfig, context: CalculationContext) -> None:
        """Check calculation-specific configuration."""

    @abstractmethod
    def estimate_resources(
        self,
        config: AppConfig,
        context: CalculationContext,
    ) -> ResourceEstimate:
        """Estimate resources without running the full calculation."""
        ...

    @abstractmethod
    def run(
        self,
        config: AppConfig,
        context: CalculationContext,
        executor: Executor,
    ) -> int:
        """Execute the calculation using the selected backend."""
        ...
