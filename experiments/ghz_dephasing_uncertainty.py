import csv
import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to find qmeasure package (for repo-local runs).
sys.path.insert(0, str(Path(__file__).parent.parent))

from qmeasure.measure import measure_projective
from qmeasure.metrics import wilson_ci
from qmeasure.noise import kraus_dephasing

OUT_DIR = Path("outputs/ghz_dephasing_uncertainty")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def H() -> np.ndarray:
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def I2() -> np.ndarray:
    return np.eye(2, dtype=np.complex128)


def kron(*mats: np.ndarray) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def ket0(n: int) -> np.ndarray:
    v = np.zeros((2**n,), dtype=np.complex128)
    v[0] = 1.0 + 0j
    return v


def rho_from_ket(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())


def apply_unitary_rho(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U @ rho @ U.conj().T


def cnot(n: int, control: int, target: int) -> np.ndarray:
    # Build full unitary for CNOT in n-qubit computational basis.
    dim = 2**n
    U = np.zeros((dim, dim), dtype=np.complex128)
    for x in range(dim):
        b = list(format(x, f"0{n}b"))
        if b[control] == "1":
            b[target] = "0" if b[target] == "1" else "1"
        y = int("".join(b), 2)
        U[y, x] = 1.0
    return U


def apply_kraus_single_qubit(
    rho: np.ndarray, Ks: list[np.ndarray], n: int, target: int
) -> np.ndarray:
    # Apply single-qubit Kraus on target qubit in n-qubit system.
    out = np.zeros_like(rho, dtype=np.complex128)
    for K in Ks:
        ops = []
        for i in range(n):
            ops.append(K if i == target else I2())
        K_full = kron(*ops)
        out += K_full @ rho @ K_full.conj().T
    return out


def even_parity_mass(hist: dict[str, int], shots: int) -> float:
    # After H on all qubits, coherence shows up in parity structure.
    total = 0
    for bitstr, c in hist.items():
        if bitstr.count("1") % 2 == 0:
            total += c
    return total / shots


def main() -> None:
    n = 3
    seed = 42

    # Build GHZ state: H on qubit0 then CNOT 0->1 and 0->2.
    ket = ket0(n)
    U_H0 = kron(H(), I2(), I2())
    rho = rho_from_ket(U_H0 @ ket)
    rho = apply_unitary_rho(rho, cnot(n, 0, 1))
    rho = apply_unitary_rho(rho, cnot(n, 0, 2))

    # Sweep dephasing and shot counts to show uncertainty grows.
    ps = [0.0, 0.02, 0.05, 0.10]
    shots_list = [200, 1000, 5000, 20000]

    rows: list[dict[str, float | int]] = []
    payload: dict[str, object] = {
        "experiment": "ghz_dephasing_uncertainty",
        "n_qubits": n,
        "ps": ps,
        "shots_list": shots_list,
        "results": [],
    }

    for p in ps:
        rho_p = rho.copy()
        Ks = kraus_dephasing(p)
        for tq in range(n):
            rho_p = apply_kraus_single_qubit(rho_p, Ks, n=n, target=tq)

        # Measure in X basis: apply H to all qubits before measurement.
        U_H_all = kron(H(), H(), H())
        rho_x = apply_unitary_rho(rho_p, U_H_all)

        for shots in shots_list:
            hist = measure_projective(
                rho_x, shots=shots, seed=seed + int(1e6 * p) + shots
            )
            m = even_parity_mass(hist, shots)
            k = int(round(m * shots))  # CI approximation for parity mass
            lo, hi = wilson_ci(k, shots)

            row = {
                "dephasing_p": float(p),
                "shots": int(shots),
                "even_parity_mass_hat": float(m),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "hist_000": int(hist.get("000", 0)),
                "hist_111": int(hist.get("111", 0)),
            }
            rows.append(row)
            payload["results"].append(row)  # type: ignore[index]

    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dephasing_p",
                "shots",
                "even_parity_mass_hat",
                "ci_lo",
                "ci_hi",
                "hist_000",
                "hist_111",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    (OUT_DIR / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    report = f"""# GHZ Dephasing Uncertainty

Tracks even parity mass (after H on all qubits) under dephasing.

- qubits: {n}
- dephasing p values: {ps}
- shots: {shots_list}

Expected:
- As p increases, coherence degrades and the even parity mass tends to drop.
- As shots decrease, CI bands widen.

Outputs:
- results.csv
- results.json
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print("Done.")
    print(f"Saved: {csv_path}")
    print(f"Saved: {OUT_DIR / 'results.json'}")
    print(f"Saved: {OUT_DIR / 'report.md'}")
    print("Sample rows:", rows[:4])


if __name__ == "__main__":
    main()
