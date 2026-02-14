from dataclasses import dataclass, field
from typing import Any


@dataclass
class NoiseSpec:
    kind: str  # "depolarizing" | "dephasing" | "amplitude_damping" | "readout"
    strength: float  # e.g. p or gamma
    targets: list[int] | None = None  # qubits affected


@dataclass
class MeasurementSpec:
    kind: str  # "projective"
    basis: str = "computational"
    targets: list[int] | None = None  # None => all qubits


@dataclass
class ExperimentConfig:
    name: str
    n_qubits: int
    shots: int = 10_000
    seed: int = 42
    noise: list[NoiseSpec] | None = None
    measurement: MeasurementSpec = field(
        default_factory=lambda: MeasurementSpec(kind="projective")
    )
    meta: dict[str, Any] | None = None
