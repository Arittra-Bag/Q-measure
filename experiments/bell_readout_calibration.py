import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to find qmeasure package (for repo-local runs).
sys.path.insert(0, str(Path(__file__).parent.parent))

from qmeasure.calibrate import fit_readout_bitflip_independent
from qmeasure.config import ExperimentConfig, MeasurementSpec
from qmeasure.measure import apply_readout_error
from qmeasure.report import make_markdown_report, save_results
from qmeasure.simulate import simulate_density

OUT_DIR = Path("outputs/bell_readout_calibration")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    shots = 20_000
    seed = 42

    # Ground truth readout noise we will inject (unknown to fitter)
    true_p01 = 0.06
    true_p10 = 0.03

    # --- Circuit: Bell state via unitaries ---
    def H() -> np.ndarray:
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

    def I2() -> np.ndarray:
        return np.eye(2, dtype=np.complex128)

    def CNOT_2q() -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.complex128,
        )

    def kron(*mats: np.ndarray) -> np.ndarray:
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def bell_circuit(n_qubits: int):
        if n_qubits != 2:
            raise ValueError("bell_circuit requires n_qubits=2")
        U1 = kron(H(), I2())  # H on qubit 0
        U2 = CNOT_2q()  # CNOT 0->1
        return [U1, U2]

    cfg = ExperimentConfig(
        name="bell_readout_calibration",
        n_qubits=2,
        shots=shots,
        seed=seed,
        noise=None,
        measurement=MeasurementSpec(kind="projective"),
    )

    out = simulate_density(cfg, circuit_fn=bell_circuit)
    ideal_hist = out["histogram"]

    # Inject readout error (ground-truth).
    noisy_hist = apply_readout_error(
        ideal_hist, p01=true_p01, p10=true_p10, seed=seed + 1
    )

    ideal_probs = {"00": 0.5, "11": 0.5, "01": 0.0, "10": 0.0}
    fit = fit_readout_bitflip_independent(
        observed_hist=noisy_hist, ideal_probs=ideal_probs, metric="tvd"
    )

    payload = {
        "experiment": "bell_readout_calibration",
        "config": {"n_qubits": 2, "shots": shots, "seed": seed},
        "true_readout": {"p01": true_p01, "p10": true_p10},
        "fit": fit,
        "ideal_hist": ideal_hist,
        "noisy_hist": noisy_hist,
        "histogram": noisy_hist,
    }

    save_results(str(OUT_DIR), payload)
    (OUT_DIR / "report.md").write_text(make_markdown_report(payload), encoding="utf-8")

    best = fit["best"]
    print("Done.")
    print(f"Saved: {OUT_DIR / 'results.json'}")
    print(f"Saved: {OUT_DIR / 'report.md'}")
    print(
        f"True p01={true_p01:.3f}, p10={true_p10:.3f} | "
        f"Fit p01={best['p01']:.3f}, p10={best['p10']:.3f} | score={best['score']:.5f}"
    )


if __name__ == "__main__":
    main()
