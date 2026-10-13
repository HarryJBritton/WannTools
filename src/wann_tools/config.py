from dataclasses import dataclass, field
from doctest import debug
from gc import enable
from pathlib import Path
from typing import Any, Literal, Union, Type
from webbrowser import get

import numpy as np

import yaml


KSpaceType = Literal['path', 'surface', 'volume']
CalculationType = Literal['spectral']
BackendType = Literal['serial', 'mpi']


@dataclass
class FiniteHamiltonianConfig:
    enabled: bool = False
    layers: int | None = None

    def validate(self) -> None:
        if self.enabled:
            if self.layers is None or self.layers < 1:
                raise ValueError(
                    'hamiltonian.finite.layers must be a positive integer  '
                    'when finite.enabled is true'
                )

@dataclass
class HamiltonianConfig:
    hr_file: Path
    nnkp_file: Path
    dtype: int = 64
    finite: FiniteHamiltonianConfig = field(
        default_factory=FiniteHamiltonianConfig
    )

    def validate(self) -> None:
        if not str(self.hr_file):
            raise ValueError("hamiltonian.hr_file is required")
        if not str(self.nnkp_file):
            raise ValueError("hamiltonian.nnkp_file is required")
        if self.dtype not in (64,128):
            raise ValueError(f"Invalid complex datatype: {self.dtype}. It must be either 64 or 128.")
        self.finite.validate()



@dataclass
class PathConfig:
    path: list[str]
    nps: int = 100

    def validate(self) -> None:
        if len(self.path) < 2:
            raise ValueError(
                "kspace.path.path must contain at least two points"
            )
        if self.nps < 1:
            raise ValueError("nps must be positive")


@dataclass
class SurfaceConfig:
    n1: int = 50
    n2: int = 50
    level: float = 0
    fixed_axis: int = 2

    def validate(self) -> None:
        if self.n1 < 1:
            raise ValueError("surface.n1 must be positive")        
        if self.n2 < 1:
            raise ValueError("surface.n2 must be positive")
        if self.fixed_axis not in (0,1,2):
            raise ValueError("surface.fixed_axis must be in (0, 1, 2)")


@dataclass
class VolumeConfig:
    nk: list[int] = field(
        default_factory=lambda: [50, 50, 50]
    )

    def validate(self) -> None:
        if len(self.nk) != 3 or any(
            n < 1 for n in self.nk
        ):
            raise ValueError(
                "volume.nk must contain three positive integers"
            )


@dataclass
class KSpaceConfig:
    type: KSpaceType
    path: PathConfig | None = None
    surface: SurfaceConfig | None = None
    volume: VolumeConfig | None = None

    def validate(self, finite_enabled: bool) -> None:
        if self.type == "path":
            if self.path is None:
                raise ValueError("kspace.path settings are required")
            self.path.validate()

        elif self.type == "surface":
            if self.surface is None:
                raise ValueError("kspace.surface settings are required")
            self.surface.validate()

        elif self.type == "volume":
            if self.volume is None:
                raise ValueError("kspace.volume settings are required")
            self.volume.validate()

        else:
            raise ValueError(f"Unsupported k-space type: {self.type!r}")

        if self.type == "volume" and finite_enabled is True:
            raise ValueError(
                f'k-space type {self.type} is not allowed with finite_enabled == True'
            )


# ====================
# Calcualtion Configs
# ====================

# TODO: allow for energy range for when path is used

@dataclass
class OmegaConfig:
    mode: str = "levels"
    levels: list[float] = field(default_factory=list)
    lower: float | None = None
    upper: float | None = None
    steps: int | None = None

    def validate(self):
        if self.mode == "levels":
            if not self.levels:
                raise ValueError(
                    "omega.levels must contain at least one energy."
                )

        elif self.mode == "range":
            if self.lower is None or self.upper is None:
                raise ValueError(
                    "omega.lower and omega.upper are required "
                    "when mode='range'."
                )

            if self.upper < self.lower:
                raise ValueError(
                    "omega.upper must be greater than or equal "
                    "to omega.lower."
                )

            if self.steps is None or self.steps < 2:
                raise ValueError(
                    "omega.steps must be at least 2."
                )

        else:
            raise ValueError(
                "omega.mode must be 'levels' or 'range'."
            )

    def to_array(self):
        """Return the configured energies as a NumPy array."""
        import numpy as np

        self.validate()

        if self.mode == "levels":
            return np.asarray(self.levels, dtype=float)

        return np.linspace(
            self.lower,
            self.upper,
            self.steps,
        )

@dataclass
class SpectralConfig:
    omega: OmegaConfig = field(default_factory=OmegaConfig)
    eta: float = 0.01
    spin_projection: str | None = None
    layer_projection: int | None = None

    def validate(self, layers: int | None) -> None:

        self.omega.validate()
        
        if self.eta <= 0:
            raise ValueError("spectral.eta must be positive")
        
        if self.spin_projection not in (None, "0", "x", "y", "z"):
            raise ValueError(
                "spin_projection must be one of: null, '0', 'x', 'y', 'z'"
            )
        
        if self.layer_projection is not None:
            if self.layer_projection < 0:
                raise ValueError("layer_projection cannot be negative")
            if layers is not None and self.layer_projection >= layers:
                raise ValueError(
                    "layer_projection must be smaller than the layer count"
                )


@dataclass
class CalculationConfig:
    type: CalculationType
    spectral: SpectralConfig | None = None

    def validate(self, layers: int | None) -> None:
        if self.type == "spectral":
            if self.spectral is None:
                raise ValueError(
                    "calculation.spectral settings are required"
                )
            self.spectral.validate(layers)
        else:
            raise ValueError(
                f"Unsupported calculation type: {self.type!r}"
            )


@dataclass
class ExecutionConfig:
    backend: BackendType = "serial"
    ranks: int = 1
    chunk_size: int = 8
    memory_limit_gb: float | None = 16
    time_limit_hours: float | None = None

    def validate(self) -> None:
        if self.backend not in ("serial", "mpi"):
            raise ValueError(f"Unsupported backend: {self.backend!r}")
        if self.ranks < 1:
            raise ValueError("execution.ranks must be positive")
        if self.backend == "serial" and self.ranks != 1:
            raise ValueError("Serial execution requires ranks: 1")
        if self.chunk_size < 1:
            raise ValueError("execution.chunk_size must be positive")
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
        if self.time_limit_hours is not None and self.time_limit_hours <= 0:
            raise ValueError("time_limit_hours must be positive")


@dataclass
class OutputConfig:
    directory: Path = Path("data/out")
    save_data: bool = True
    plot: bool = True
    format: str = "pdf"


@dataclass
class DebugConfig:
    enabled: bool = False
    log_level: str = "INFO"
    log_chunks: bool = True
    log_chunk_timing: bool = True
    limit_chunks: int | None = None

    def validate(self) -> None:
        if self.log_level.upper() not in {
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        }:
            raise ValueError(f"Invalid log level: {self.log_level}")

        if self.limit_chunks is not None and self.limit_chunks < 1:
            raise ValueError("debug.limit_chunks must be positive")

        
@dataclass
class AppConfig:
    hamiltonian: HamiltonianConfig
    kspace: KSpaceConfig
    calculation: CalculationConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def validate(self) -> None:
        self.hamiltonian.validate()

        layers = (
            self.hamiltonian.finite.layers
            if self.hamiltonian.finite.enabled
            else None
        )

        self.kspace.validate(
            finite_enabled=self.hamiltonian.finite.enabled
        )
        self.calculation.validate(layers)
        self.execution.validate()
        self.debug.validate()


def _build_config(raw: dict[str, Any]) -> AppConfig:
    """Convert parsed YAML dictionaries into typed config objects."""
    try:
        h_raw = raw["hamiltonian"]
        finite_raw = h_raw.get("finite", {})

        k_raw = raw["kspace"]
        c_raw = raw["calculation"]
        spectral_raw = c_raw.get('spectral', {})
        e_raw = raw.get("execution", {})
        o_raw = raw.get("output", {})

        hamiltonian = HamiltonianConfig(
            hr_file=Path(h_raw["hr_file"]),
            nnkp_file=Path(h_raw["nnkp_file"]),
            dtype=h_raw["dtype"],
            finite=FiniteHamiltonianConfig(**finite_raw),
        )

        path = (
            PathConfig(**k_raw["path"])
            if "path" in k_raw else None
        )
        surface = (
            SurfaceConfig(**k_raw["surface"])
            if "surface" in k_raw else None
        )
        volume = (
            VolumeConfig(**k_raw["volume"])
            if "volume" in k_raw else None
        )
        kspace = KSpaceConfig(
            type=k_raw["type"],
            path=path,
            surface=surface,
            volume=volume,
        )

        spectral = (
            SpectralConfig(
                omega=OmegaConfig(**spectral_raw['omega']),
                eta=spectral_raw.get('eta', 0.01),
                spin_projection=spectral_raw.get("spin_projection"),
                layer_projection=spectral_raw.get("layer_projection"),

            )
            if "spectral" in c_raw
            else None
        )
        calculation = CalculationConfig(
            type=c_raw["type"],
            spectral=spectral,
        )

        execution = ExecutionConfig(**e_raw)
        if "directory" in o_raw:
            o_raw["directory"] = Path(o_raw["directory"])
        output = OutputConfig(**o_raw)

        debug = DebugConfig(**raw.get("debug", {}))

        config = AppConfig(
            hamiltonian=hamiltonian,
            kspace=kspace,
            calculation=calculation,
            execution=execution,
            output=output,
            debug=debug,
        )
        config.validate()
        return config

    except KeyError as exc:
        raise ValueError(
            f"Missing required configuration field: {exc}"
        ) from exc
    except TypeError as exc:
        raise ValueError(
            f"Invalid configuration fields or value types: {exc}"
        ) from exc




def load_config(filename: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file."""
    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    if not isinstance(raw, dict):
        raise ValueError("The configuration file must contain a YAML mapping")

    return _build_config(raw)