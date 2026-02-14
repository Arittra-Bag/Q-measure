import numpy as np

from qmeasure.core import apply_unitary, density_from_ket, ket0


def test_ket0_shape_and_norm():
    v = ket0(3)
    assert v.shape == (8,)
    assert np.isclose(np.vdot(v, v).real, 1.0)
    assert v[0] == 1.0 + 0j
    assert np.all(v[1:] == 0.0 + 0j)


def test_density_from_ket_is_projector():
    v = np.array([1.0, 0.0], dtype=np.complex128)
    rho = density_from_ket(v)
    assert rho.shape == (2, 2)
    assert np.allclose(rho, np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128))


def test_apply_unitary_ket_and_rho():
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    v0 = np.array([1.0, 0.0], dtype=np.complex128)
    v1 = apply_unitary(v0, X, kind="ket")
    assert np.allclose(v1, np.array([0.0, 1.0], dtype=np.complex128))

    rho0 = density_from_ket(v0)
    rho1 = apply_unitary(rho0, X, kind="rho")
    assert np.allclose(rho1, np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128))
