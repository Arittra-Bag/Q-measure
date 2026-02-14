import numpy as np

from qmeasure.measure import (
    apply_readout_error,
    apply_readout_error_probs,
    measure_projective,
)


def test_apply_readout_error_preserves_total_counts():
    hist = {"0": 10, "1": 7}
    out = apply_readout_error(hist, p01=0.2, p10=0.3, seed=123)
    assert sum(out.values()) == sum(hist.values())


def test_apply_readout_error_probs_normalizes_and_expands_space():
    probs = {"00": 0.5, "11": 0.5}
    out = apply_readout_error_probs(probs, p01=0.1, p10=0.2)
    assert set(out.keys()) == {"00", "01", "10", "11"}
    assert abs(sum(out.values()) - 1.0) < 1e-12
    assert all(v >= 0.0 for v in out.values())


def test_measure_projective_simple_pure_state():
    # |0><0| should always measure "0" in computational basis.
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    hist = measure_projective(rho, shots=50, seed=0)
    assert hist == {"0": 50}
