from collections.abc import Callable

import numpy as np

Kraus = list[np.ndarray]


def kraus_depolarizing(p: float) -> Kraus:
    """
    Single-qubit depolarizing channel:
      E(rho) = (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)
    """
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")
    Id = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    K0 = np.sqrt(1 - p) * Id
    K1 = np.sqrt(p / 3.0) * X
    K2 = np.sqrt(p / 3.0) * Y
    K3 = np.sqrt(p / 3.0) * Z
    return [K0, K1, K2, K3]


def kraus_dephasing(p: float) -> Kraus:
    """
    Single-qubit dephasing / phase-flip:
      E(rho) = (1-p) rho + p Z rho Z
    """
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")
    Id = np.eye(2, dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    K0 = np.sqrt(1 - p) * Id
    K1 = np.sqrt(p) * Z
    return [K0, K1]


def kraus_amplitude_damping(gamma: float) -> Kraus:
    """
    Single-qubit amplitude damping with parameter gamma in [0,1].
    """
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError("gamma must be in [0,1]")
    K0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=np.complex128)
    return [K0, K1]


def apply_kraus(rho: np.ndarray, K: Kraus) -> np.ndarray:
    """
    Apply Kraus operators to a density matrix:
      rho' = sum_k K_k rho K_k^dagger
    """
    out = np.zeros_like(rho, dtype=np.complex128)
    for A in K:
        out += A @ rho @ A.conj().T
    return out


def compose_channels(
    channels: list[Callable[[np.ndarray], np.ndarray]],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Compose a list of channel functions: f(g(h(rho))).
    """

    def _f(rho: np.ndarray) -> np.ndarray:
        out = rho
        for ch in channels:
            out = ch(out)
        return out

    return _f
