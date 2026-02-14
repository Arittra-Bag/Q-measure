import numpy as np

from qmeasure.config import ExperimentConfig, MeasurementSpec, NoiseSpec
from qmeasure.simulate import simulate_density, simulate_mc


def _I(dim: int) -> np.ndarray:
    return np.eye(dim, dtype=np.complex128)


def test_simulate_density_identity_circuit_1q():
    def circuit_fn(n: int):
        assert n == 1
        return _I(2)

    cfg = ExperimentConfig(
        name="id_1q",
        n_qubits=1,
        shots=100,
        seed=1,
        noise=None,
        measurement=MeasurementSpec(kind="projective"),
    )
    out = simulate_density(cfg, circuit_fn=circuit_fn)
    assert out["histogram"] == {"0": 100}


def test_simulate_mc_identity_circuit_1q():
    def circuit_fn(n: int):
        assert n == 1
        return _I(2)

    cfg = ExperimentConfig(
        name="id_1q_mc",
        n_qubits=1,
        shots=50,
        seed=2,
        noise=None,
        measurement=MeasurementSpec(kind="projective"),
    )
    out = simulate_mc(cfg, circuit_fn=circuit_fn, n_trajectories=10, ci_alpha=0.1)
    mc = out["mc"]
    assert set(mc["mean_probs"].keys()) == {"0", "1"}
    assert 0.0 <= mc["mean_probs"]["0"] <= 1.0
    assert 0.0 <= mc["mean_probs"]["1"] <= 1.0
    assert abs(mc["mean_probs"]["0"] + mc["mean_probs"]["1"] - 1.0) < 1e-12
    assert sum(out["histogram"].values()) == 50


def test_simulate_mc_with_readout_noise_keeps_probabilities_valid():
    def circuit_fn(n: int):
        assert n == 1
        return _I(2)

    cfg = ExperimentConfig(
        name="id_1q_ro",
        n_qubits=1,
        shots=200,
        seed=3,
        noise=[NoiseSpec(kind="readout", strength=0.25)],
        measurement=MeasurementSpec(kind="projective"),
        meta={"p10": 0.10},
    )
    out = simulate_mc(cfg, circuit_fn=circuit_fn, n_trajectories=20, ci_alpha=0.2)
    mc = out["mc"]
    for k in ["0", "1"]:
        assert 0.0 <= mc["p_lo_probs"][k] <= 1.0
        assert 0.0 <= mc["p_hi_probs"][k] <= 1.0
