import numpy as np

from qmeasure.noise import (
    apply_kraus,
    kraus_amplitude_damping,
    kraus_dephasing,
    kraus_depolarizing,
)


def _kraus_completeness(Ks):
    acc = np.zeros((2, 2), dtype=np.complex128)
    for K in Ks:
        acc += K.conj().T @ K
    return acc


def _random_density(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(2,)) + 1j * rng.normal(size=(2,))
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def test_kraus_completeness_dephasing():
    Ks = kraus_dephasing(0.3)
    C = _kraus_completeness(Ks)
    assert np.allclose(C, np.eye(2), atol=1e-12)


def test_kraus_completeness_depolarizing():
    Ks = kraus_depolarizing(0.2)
    C = _kraus_completeness(Ks)
    assert np.allclose(C, np.eye(2), atol=1e-12)


def test_kraus_completeness_amplitude_damping():
    Ks = kraus_amplitude_damping(0.4)
    C = _kraus_completeness(Ks)
    assert np.allclose(C, np.eye(2), atol=1e-12)


def test_apply_kraus_preserves_trace():
    rho = _random_density(seed=123)
    for Ks in [
        kraus_dephasing(0.1),
        kraus_depolarizing(0.2),
        kraus_amplitude_damping(0.3),
    ]:
        rho2 = apply_kraus(rho, Ks)
        assert np.allclose(np.trace(rho2), 1.0 + 0j, atol=1e-12)
