import numpy as np

from qmeasure.calibrate import (
    fit_noise_strength_grid,
    fit_readout_bitflip_independent,
    fit_readout_error_1q,
)
from qmeasure.measure import apply_readout_error_probs
from qmeasure.report import make_markdown_report, save_results


def test_fit_readout_error_1q_recovers_rates():
    # True p01 = 0.2, p10 = 0.1 on fixed counts.
    confusion = {
        "00": 80,  # true 0 observed 0
        "01": 20,  # true 0 observed 1
        "10": 10,  # true 1 observed 0
        "11": 90,  # true 1 observed 1
    }
    fit = fit_readout_error_1q(confusion)
    assert abs(fit["p01"] - 0.2) < 1e-12
    assert abs(fit["p10"] - 0.1) < 1e-12


def test_fit_readout_bitflip_independent_grid_hits_true_params():
    ideal_probs = {"00": 0.5, "11": 0.5, "01": 0.0, "10": 0.0}
    true_p01 = 0.05
    true_p10 = 0.10

    obs_probs = apply_readout_error_probs(ideal_probs, p01=true_p01, p10=true_p10)
    shots = 200_000
    observed_hist = {k: int(round(v * shots)) for k, v in obs_probs.items()}

    fit = fit_readout_bitflip_independent(observed_hist, ideal_probs, metric="tvd")
    best = fit["best"]
    assert best["p01"] == true_p01
    assert best["p10"] == true_p10


def test_fit_noise_strength_grid_selects_best():
    # 1-qubit synthetic: P(1) = strength, P(0)=1-strength.
    g_true = 0.3
    observed = {"0": 700, "1": 300}
    grid = np.linspace(0.0, 1.0, 11)

    def simulate_fn(g: float):
        g = float(g)
        return {"0": int(round(1000 * (1.0 - g))), "1": int(round(1000 * g))}

    out = fit_noise_strength_grid(
        observed, grid=grid, simulate_fn=simulate_fn, metric="tvd"
    )
    assert abs(out["best_strength"] - g_true) < 1e-12


def test_report_save_and_markdown(tmp_path):
    payload = {
        "experiment": "test",
        "config": {"n_qubits": 1, "shots": 10},
        "histogram": {"0": 10, "1": 0},
        "results": [{"x": 1, "y": 2}, {"x": 2, "y": 3}],
    }
    out_dir = tmp_path / "out"
    p = save_results(str(out_dir), payload)
    assert (out_dir / "results.json").exists()
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "histogram.csv").exists()
    assert p.endswith("results.json")

    md = make_markdown_report(payload)
    assert "# test" in md.lower() or "# Test" in md
