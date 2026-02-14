from typing import Literal

import numpy as np

StateKind = Literal["ket", "rho"]


def ket0(n_qubits: int) -> np.ndarray:
    """|0...0> state vector, shape (2^n,)."""
    n = int(n_qubits)
    if n < 1:
        raise ValueError("n_qubits must be >= 1")
    v = np.zeros((2**n,), dtype=np.complex128)
    v[0] = 1.0 + 0j
    return v


def density_from_ket(ket: np.ndarray) -> np.ndarray:
    """rho = |psi><psi|."""
    ket = np.asarray(ket, dtype=np.complex128)
    if ket.ndim != 1:
        raise ValueError("ket must be a 1D state vector")

    norm2 = float(np.vdot(ket, ket).real)
    if norm2 <= 0.0:
        raise ValueError("ket has non-positive norm")

    # Allow mild numerical drift; renormalize only if close.
    if not np.isclose(norm2, 1.0, atol=1e-8, rtol=0.0):
        if np.isclose(norm2, 1.0, atol=1e-3, rtol=0.0):
            ket = ket / np.sqrt(norm2)
        else:
            raise ValueError(f"ket is not normalized (||ket||^2={norm2})")

    return np.outer(ket, ket.conj())


def apply_unitary(state: np.ndarray, U: np.ndarray, kind: StateKind) -> np.ndarray:
    """Apply U to ket or rho."""
    U = np.asarray(U, dtype=np.complex128)
    state = np.asarray(state, dtype=np.complex128)

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be a square matrix")

    if kind == "ket":
        if state.ndim != 1 or state.shape[0] != U.shape[0]:
            raise ValueError("ket state must have shape (dim,) matching U")
        return np.asarray(U @ state, dtype=np.complex128)

    if kind == "rho":
        if state.ndim != 2 or state.shape != U.shape:
            raise ValueError("rho state must have shape (dim, dim) matching U")
        return np.asarray(U @ state @ U.conj().T, dtype=np.complex128)

    raise ValueError("kind must be 'ket' or 'rho'")
