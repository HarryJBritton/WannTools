
from wann_tools.config import AppConfig, ExecutionConfig

from dataclasses import dataclass, field
import math

from wann_tools.calculations.base import ResourceEstimate


from wann_tools.wannham import build_hamiltonian
from wann_tools.kspace import build_kspace
from wann_tools.calculations.registry import get_calculation
from wann_tools.context import build_context
from wann_tools.executors import build_executor

@dataclass
class ExecutionPlan:
    calculation_estimate: ResourceEstimate

    chunk_size: int
    total_chunks: int
    active_workers: int

    estimated_peak_memory_bytes_per_worker: int
    estimated_peak_memory_bytes_per_job: int

    memory_limit_bytes_per_worker: int | None

    warnings: list[str] = field(default_factory=list)


    @property
    def fits_memory_limit(self) -> bool | None :
        if self.memory_limit_bytes_per_worker is None:
            return None

        return (
            self.estimated_peak_memory_bytes_per_worker 
            <= self.memory_limit_bytes_per_worker
        )


class ResourcePlanner:
    def plan(self, estimate: ResourceEstimate, execution: ExecutionConfig) -> ExecutionPlan:
        if execution.chunk_size < 1:
            raise ValueError("execution.chunk_size must be positive.")

        if estimate.total_work_items < 0:
            raise ValueError("total_work_items cannot be negative.")

        if estimate.peak_memory_bytes_per_worker < 0:
            raise ValueError(
                "peak_memory_bytes_per_worker cannot be negative."
            )

        backend = execution.backend.lower()

        if backend == "serial":
            active_workers = 1
        elif backend == "mpi":
            if execution.ranks < 1:
                raise ValueError("execution.ranks must be positive.")
            active_workers = execution.ranks
        else:
            raise ValueError(f"Unsupported backend: {backend!r}")

        total_chunks = math.ceil(
            estimate.total_work_items / execution.chunk_size
        )

        # Assumes each worker processes one chunk at a time.
        # Calculation-specific estimates should ideally already
        # reflect the configured chunk size.
        per_worker_bytes = estimate.peak_memory_bytes_per_worker

        # Initial approximation: each active worker has its own
        # working set. Shared memory and root-only output storage
        # need additional modelling if relevant.
        job_bytes = per_worker_bytes * min(
            active_workers,
            total_chunks,
        )

        limit_bytes = None
        if execution.memory_limit_gb is not None:
            limit_bytes = int(execution.memory_limit_gb * 1024**3)

        warnings = list(estimate.assumptions or [])

        if (
            limit_bytes is not None
            and job_bytes > limit_bytes
        ):
            warnings.append(
                "Estimated per-worker memory exceeds the configured "
                "memory limit."
            )

        return ExecutionPlan(
            calculation_estimate=estimate,
            chunk_size=execution.chunk_size,
            total_chunks=total_chunks,
            active_workers=active_workers,
            estimated_peak_memory_bytes_per_worker=per_worker_bytes,
            estimated_peak_memory_bytes_per_job=job_bytes,
            memory_limit_bytes_per_worker=limit_bytes,
            warnings=warnings,
        )


def estimate(config: AppConfig):
    ham = build_hamiltonian(config)
    kspace = build_kspace(config)

    calculation = get_calculation(config.calculation.type)

    context = build_context(config, ham, kspace)

    calculation.validate(
        config=config,
        context=context,
    )


    res_estimate = calculation.estimate_resources(
        config,
        context
    )

    plan = ResourcePlanner().plan(
        estimate=res_estimate,
        execution=config.execution,
    )

    print(f"Work items: {res_estimate.total_work_items:,}")
    print(f"Chunks: {plan.total_chunks:,}")
    print(f"Active workers: {plan.active_workers}")
    print(
        "Estimated peak memory per worker: "
        f"{plan.estimated_peak_memory_bytes_per_worker / 1024**3:.2f} GiB"
    )
    print(
        "Estimated peak memory for the job: "
        f"{plan.estimated_peak_memory_bytes_per_job / 1024**3:.2f} GiB"
    )

    for warning in plan.warnings:
        print(f"WARNING: {warning}")

    if plan.fits_memory_limit is False:
        raise MemoryError(
            "The estimated per-worker memory exceeds the configured limit."
        )
