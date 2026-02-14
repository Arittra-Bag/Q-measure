from collections.abc import Callable
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .measure import apply_readout_error, measure_projective
from .noise import (
    kraus_amplitude_damping,
    kraus_dephasing,
    kraus_depolarizing,
)


def _I() -> np.ndarray:
    return np.eye(2, dtype=np.complex128)


def _kron(*mats: np.ndarray) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _ket0(n: int) -> np.ndarray:
    v = np.zeros((2**n,), dtype=np.complex128)
    v[0] = 1.0 + 0j
    return v


def _rho_from_ket(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())


def _apply_unitary_rho(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    return np.asarray(U @ rho @ U.conj().T, dtype=np.complex128)


def _apply_single_qubit_kraus(
    rho: np.ndarray, Ks: list[np.ndarray], n: int, target: int
) -> np.ndarray:
    """
    Apply a 1-qubit Kraus set Ks on 'target' qubit of an n-qubit density matrix.
    """
    out = np.zeros_like(rho, dtype=np.complex128)
    for K in Ks:
        ops = []
        for i in range(n):
            ops.append(K if i == target else _I())
        K_full = _kron(*ops)
        out += K_full @ rho @ K_full.conj().T
    return out


def _resolve_targets(n_qubits: int, targets: list[int] | None) -> list[int]:
    if targets is None:
        return list(range(n_qubits))
    return [int(t) for t in targets]


def _all_bitstrings(n: int) -> list[str]:
    return [format(i, f"0{n}b") for i in range(2**n)]


def _hist_to_prob_vec(hist: dict[str, int], keys: list[str]) -> np.ndarray:
    total = int(sum(hist.values()))
    if total <= 0:
        return np.zeros((len(keys),), dtype=float)
    return np.array([hist.get(k, 0) / total for k in keys], dtype=float)


def _build_rho_from_circuit_fn(n: int, circuit_fn: Callable[[int], Any]) -> np.ndarray:
    """
    Contract for circuit_fn (MVP):
      - circuit_fn(n_qubits) can return:
          (A) a unitary U of shape (2^n, 2^n), OR
          (B) a list of unitaries [U1, U2, ...], OR
          (C) a prepared density matrix rho of shape (2^n, 2^n)
    """
    obj = circuit_fn(n)

    if isinstance(obj, list):
        rho = _rho_from_ket(_ket0(n))
        for U in obj:
            rho = _apply_unitary_rho(rho, np.asarray(U))
        return rho

    arr = np.asarray(obj)
    if arr.ndim == 2 and arr.shape == (2**n, 2**n):
        # Could be either a unitary or rho; detect via a unitary check.
        UU = arr @ arr.conj().T
        if np.allclose(UU, np.eye(2**n), atol=1e-6):
            rho = _rho_from_ket(_ket0(n))
            return _apply_unitary_rho(rho, arr)
        return arr.astype(np.complex128)

    raise ValueError(
        "circuit_fn must return rho (2^n x 2^n), unitary (2^n x 2^n) or list of unitaries."
    )


def _apply_noise_specs(rho: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    """
    Deterministic noise application to a density matrix.

    Note: for MVP, noise is applied once after circuit preparation.
    """
    n = int(config.n_qubits)
    if not config.noise:
        return rho

    for ns in config.noise:
        kind = ns.kind
        strength = float(ns.strength)
        targets = _resolve_targets(n, ns.targets)

        if kind == "dephasing":
            Ks = kraus_dephasing(strength)
            for t in targets:
                rho = _apply_single_qubit_kraus(rho, Ks, n=n, target=t)

        elif kind == "depolarizing":
            Ks = kraus_depolarizing(strength)
            for t in targets:
                rho = _apply_single_qubit_kraus(rho, Ks, n=n, target=t)

        elif kind == "amplitude_damping":
            Ks = kraus_amplitude_damping(strength)
            for t in targets:
                rho = _apply_single_qubit_kraus(rho, Ks, n=n, target=t)

        elif kind == "readout":
            # handled at measurement time via apply_readout_error
            pass
        else:
            raise ValueError(f"Unknown noise kind: {kind}")

    return rho


def _maybe_apply_readout_noise(
    hist: dict[str, int], config: ExperimentConfig, seed: int
) -> dict[str, int]:
    if not config.noise:
        return hist

    out = hist
    for ns in config.noise:
        if ns.kind != "readout":
            continue

        # MVP: interpret strength as p01, and default to symmetric p10 unless supplied.
        p01 = float(ns.strength)
        p10 = float(ns.strength)
        if config.meta and "p10" in config.meta:
            p10 = float(config.meta["p10"])

        out = apply_readout_error(out, p01=p01, p10=p10, seed=seed)

    return out


def simulate_density(
    config: ExperimentConfig,
    circuit_fn,
) -> dict[str, Any]:
    """
    Deterministic simulation using density matrices + deterministic noise.

    We then sample a measurement histogram with finite shots (shot noise).
    """
    n = int(config.n_qubits)
    shots = int(config.shots)
    seed = int(config.seed)

    rho = _build_rho_from_circuit_fn(n, circuit_fn)
    rho = _apply_noise_specs(rho, config)

    hist = measure_projective(rho, shots=shots, seed=seed)
    hist = _maybe_apply_readout_noise(hist, config, seed=seed + 7)

    probs = np.real(np.diag(rho)).clip(0, 1)
    probs = probs / probs.sum()

    return {
        "rho_final": rho,
        "probs": probs,
        "histogram": hist,
        "config": {
            "name": config.name,
            "n_qubits": n,
            "shots": shots,
            "seed": seed,
        },
    }


def simulate_mc(
    config: ExperimentConfig,
    circuit_fn,
    n_trajectories: int = 200,
    ci_alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Monte Carlo uncertainty due to finite shots (and optional stochastic readout noise).

    Note: this is not Kraus-trajectory unraveling. The quantum channel noise is applied
    deterministically to the density matrix, then measurement is resampled.
    """
    n = int(config.n_qubits)
    shots = int(config.shots)
    seed = int(config.seed)

    n_traj = int(n_trajectories)
    if n_traj <= 0:
        raise ValueError("n_trajectories must be >= 1")

    alpha = float(ci_alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("ci_alpha must be in (0, 1)")

    rho = _build_rho_from_circuit_fn(n, circuit_fn)
    rho = _apply_noise_specs(rho, config)

    keys = _all_bitstrings(n)
    prob_mat = np.zeros((n_traj, len(keys)), dtype=float)

    for k in range(n_traj):
        h = measure_projective(rho, shots=shots, seed=seed + k)
        h = _maybe_apply_readout_noise(h, config, seed=(seed + 7 + k))
        prob_mat[k, :] = _hist_to_prob_vec(h, keys)

    mean_probs_vec = prob_mat.mean(axis=0)
    p_lo = alpha / 2.0
    p_hi = 1.0 - alpha / 2.0
    q_lo, q_hi = np.quantile(prob_mat, [p_lo, p_hi], axis=0)

    # Expected histogram from mean probabilities.
    counts = np.floor(mean_probs_vec * shots + 0.5).astype(int)
    diff = int(shots - counts.sum())
    if diff != 0:
        j = int(np.argmax(mean_probs_vec))
        counts[j] = max(0, int(counts[j] + diff))

    hist_mean: dict[str, int] = {k: int(c) for k, c in zip(keys, counts, strict=False)}

    probs = np.real(np.diag(rho)).clip(0, 1)
    probs = probs / probs.sum()

    return {
        "rho_final": rho,
        "probs": probs,
        "histogram": hist_mean,
        "config": {
            "name": config.name,
            "n_qubits": n,
            "shots": shots,
            "seed": seed,
        },
        "mc": {
            "n_trajectories": n_traj,
            "alpha": float(alpha),
            "mean_probs": {
                k: float(v) for k, v in zip(keys, mean_probs_vec, strict=False)
            },
            "p_lo": float(p_lo),
            "p_hi": float(p_hi),
            "p_lo_probs": {k: float(v) for k, v in zip(keys, q_lo, strict=False)},
            "p_hi_probs": {k: float(v) for k, v in zip(keys, q_hi, strict=False)},
        },
    }


def sweep_param(
    base_config: ExperimentConfig,
    param_name: str,
    values: list[float],
    runner,
    circuit_fn,
) -> dict[str, Any]:
    """
    Minimal param sweep. MVP supports:
      param_name = "noise[0].strength"
    """
    results = []
    for v in values:
        cfg = base_config
        # shallow clone (good enough for MVP)
        if cfg.noise:
            noise_copy = []
            for ns in cfg.noise:
                noise_copy.append(
                    type(ns)(kind=ns.kind, strength=ns.strength, targets=ns.targets)
                )
            cfg = type(base_config)(
                name=base_config.name,
                n_qubits=base_config.n_qubits,
                shots=base_config.shots,
                seed=base_config.seed,
                noise=noise_copy,
                measurement=base_config.measurement,
                meta=base_config.meta,
            )

        if param_name == "noise[0].strength":
            if not cfg.noise:
                raise ValueError("No noise in config to sweep.")
            cfg.noise[0].strength = float(v)
        else:
            raise ValueError("MVP sweep supports only param_name='noise[0].strength'")

        out = runner(cfg, circuit_fn)
        results.append({"value": float(v), "out": out})

    return {"param_name": param_name, "values": values, "results": results}
