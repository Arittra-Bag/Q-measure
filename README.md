# Q-MEASURE: Uncertainty and Calibration Lab for Noisy Measurement Systems

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Arittra-Bag/Q-measure?utm_source=oss&utm_medium=github&utm_campaign=Arittra-Bag%2FQ-measure&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

Q-MEASURE is a Python toolkit for simulating probabilistic state evolution under measurement collapse and configurable noise channels, with Monte Carlo uncertainty estimates and calibration routines.

> **Positioning:** This is not a quantum circuit playground.
> It is an uncertainty + noise diagnostics framework for partially observed systems.

## Why it matters
- **HFT/Quant:** partial observability, noise propagation, estimator variance, calibration
- **AI/ML:** uncertainty quantification, robustness, noisy channels, calibration under shift
- **Engineering:** reproducible experiments, report generation, clean APIs

## Features (MVP)
- Density-matrix simulation with noise channels (depolarizing, dephasing, amplitude damping)
- Measurement sampling (projective, computational basis)
- Readout error modeling (bit-flip confusion)
- Monte Carlo trajectories for uncertainty bands
- Calibration: fit readout error, fit noise strength via grid search
- Experiment runner + JSON/CSV + Markdown report outputs

## Quickstart

```python
from qmeasure.config import ExperimentConfig, NoiseSpec, MeasurementSpec
from qmeasure.simulate import simulate_density

def bell_circuit(n_qubits):
    # MVP: implement in your experiments using simple gate helpers (H, CNOT)
    ...

cfg = ExperimentConfig(
    name="bell_readout_demo",
    n_qubits=2,
    shots=20000,
    noise=[NoiseSpec(kind="dephasing", strength=0.02)],
    measurement=MeasurementSpec(kind="projective"),
)

out = simulate_density(cfg, circuit_fn=bell_circuit)
print(out["histogram"])
```

