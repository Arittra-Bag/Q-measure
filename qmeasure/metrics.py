import numpy as np


def hist_to_probs(hist: dict[str, int]) -> np.ndarray:
    """
    Convert histogram to a probability vector aligned to lexicographic bitstring order.

    Note: the returned vector is aligned to sorted(hist.keys()).
    """
    if not hist:
        return np.array([], dtype=float)
    keys = sorted(hist.keys())
    total = int(sum(hist.values()))
    if total <= 0:
        return np.zeros((len(keys),), dtype=float)
    return np.array([hist[k] / total for k in keys], dtype=float)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    KL(p||q) with epsilon smoothing.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have same shape")
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    TVD = 0.5 * sum_i |p_i - q_i|
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have same shape")
    return float(0.5 * np.sum(np.abs(p - q)))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson confidence interval for a Bernoulli proportion.
    """
    if n <= 0:
        return (0.0, 0.0)
    k = int(k)
    n = int(n)
    phat = k / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt((phat * (1.0 - phat) / n) + (z**2) / (4.0 * n**2))
    lo = max(0.0, float(center - margin))
    hi = min(1.0, float(center + margin))
    return (lo, hi)


def multinomial_ci(
    hist: dict[str, int], alpha: float = 0.05
) -> dict[str, tuple[float, float]]:
    """
    Per-outcome CI for multinomial using Wilson per outcome (MVP approximation).
    """
    if not hist:
        return {}
    total = int(sum(hist.values()))
    if total <= 0:
        return {k: (0.0, 0.0) for k in hist.keys()}

    # 1.96 ~ 95% CI (alpha=0.05). If you want alpha-driven z, we can add later.
    z = 1.96
    out: dict[str, tuple[float, float]] = {}
    for k, c in hist.items():
        out[k] = wilson_ci(int(c), int(total), z=z)
    return out
