import csv
import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to find qmeasure package (for repo-local runs).
sys.path.insert(0, str(Path(__file__).parent.parent))

from qmeasure.metrics import wilson_ci
from qmeasure.noise import apply_kraus, kraus_amplitude_damping

OUT_DIR = Path("outputs/single_qubit_damping_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rho_from_ket(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())


def ket1() -> np.ndarray:
    return np.array([0, 1], dtype=np.complex128)


def measure_prob_1(rho: np.ndarray) -> float:
    # Computational basis probability of |1> is rho[1,1].
    return float(np.real(rho[1, 1]).clip(0, 1))


def main() -> None:
    shots = 20_000
    seed = 42
    rng = np.random.default_rng(seed)

    gammas = np.linspace(0.0, 0.30, 31).tolist()

    rows: list[dict[str, float]] = []
    payload = {
        "experiment": "single_qubit_damping_sweep",
        "shots": shots,
        "seed": seed,
        "results": [],
    }

    for gamma in gammas:
        rho0 = rho_from_ket(ket1())  # start in |1>
        rho1 = apply_kraus(rho0, kraus_amplitude_damping(gamma))

        p1_true = measure_prob_1(rho1)

        samples = rng.random(shots) < p1_true
        k = int(samples.sum())
        p1_hat = k / shots
        lo, hi = wilson_ci(k, shots)

        row = {
            "gamma": float(gamma),
            "p1_true": float(p1_true),
            "p1_hat": float(p1_hat),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
        }
        rows.append(row)
        payload["results"].append(row)

    csv_path = OUT_DIR / "sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["gamma", "p1_true", "p1_hat", "ci_lo", "ci_hi"]
        )
        w.writeheader()
        w.writerows(rows)

    (OUT_DIR / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    report = f"""# Single Qubit Amplitude Damping Sweep

- Shots per gamma: {shots}
- Gamma range: {gammas[0]:.2f} .. {gammas[-1]:.2f} (N={len(gammas)})

Outputs:
- `sweep.csv` with p(|1>) estimates and Wilson CIs
- `results.json` full payload

Quick check:
- Expect p(|1>) to decrease as gamma increases.
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print("Done.")
    print(f"Saved: {csv_path}")
    print(f"Saved: {OUT_DIR / 'results.json'}")
    print(f"Saved: {OUT_DIR / 'report.md'}")
    print("Sample rows:", rows[:3])


if __name__ == "__main__":
    main()
