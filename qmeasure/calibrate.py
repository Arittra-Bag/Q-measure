from collections.abc import Callable
from typing import Any

import numpy as np

from .measure import apply_readout_error_probs
from .metrics import kl_divergence, total_variation_distance


def _infer_n_from_hist(hist: dict[str, int]) -> int:
    if not hist:
        raise ValueError("histogram is empty")
    k = next(iter(hist.keys()))
    n = len(k)
    if n < 1 or any(len(s) != n for s in hist.keys()):
        raise ValueError("histogram keys must be equal-length bitstrings")
    return n


def _infer_n_from_probs(probs: dict[str, float]) -> int:
    if not probs:
        raise ValueError("prob distribution is empty")
    k = next(iter(probs.keys()))
    n = len(k)
    if n < 1 or any(len(s) != n for s in probs.keys()):
        raise ValueError("prob keys must be equal-length bitstrings")
    return n


def _all_bitstrings(n: int) -> list[str]:
    return [format(i, f"0{n}b") for i in range(2**n)]


def _hist_to_prob_vec(hist: dict[str, int], keys: list[str]) -> np.ndarray:
    total = int(sum(hist.values()))
    if total <= 0:
        return np.zeros((len(keys),), dtype=float)
    return np.array([hist.get(k, 0) / total for k in keys], dtype=float)


def _prob_dict_to_vec(probs: dict[str, float], keys: list[str]) -> np.ndarray:
    v = np.array([float(probs.get(k, 0.0)) for k in keys], dtype=float)
    s = float(v.sum())
    if s <= 0.0:
        return np.zeros_like(v)
    return v / s


def fit_readout_error_1q(confusion_counts: dict[str, int]) -> dict[str, float]:
    """
    Fit a 1-qubit independent bit-flip readout model from a 2x2 confusion table.

    confusion_counts keys:
      "00","01","10","11" where key = true_bit + observed_bit
    """
    for k in ["00", "01", "10", "11"]:
        if k not in confusion_counts:
            raise ValueError("confusion_counts must contain keys: 00, 01, 10, 11")

    n00 = int(confusion_counts["00"])
    n01 = int(confusion_counts["01"])
    n10 = int(confusion_counts["10"])
    n11 = int(confusion_counts["11"])
    if min(n00, n01, n10, n11) < 0:
        raise ValueError("confusion counts must be non-negative")

    n0 = n00 + n01
    n1 = n10 + n11
    if n0 <= 0:
        raise ValueError("need at least one sample with true=0 (n00+n01 > 0)")
    if n1 <= 0:
        raise ValueError("need at least one sample with true=1 (n10+n11 > 0)")

    p01 = n01 / n0
    p10 = n10 / n1
    return {"p01": float(p01), "p10": float(p10), "n0": float(n0), "n1": float(n1)}


def fit_readout_bitflip_independent(
    observed_hist: dict[str, int],
    ideal_probs: dict[str, float],
    grid: np.ndarray | None = None,
    metric: str = "tvd",
) -> dict[str, Any]:
    """
    Fit independent readout bit-flip parameters p01 and p10 by grid search.
    """
    n_obs = _infer_n_from_hist(observed_hist)
    n_ideal = _infer_n_from_probs(ideal_probs)
    if n_obs != n_ideal:
        raise ValueError(
            "observed_hist and ideal_probs must have the same bitstring length"
        )

    if grid is None:
        grid = np.linspace(0.0, 0.15, 61)
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be a 1D array with at least 2 points")

    keys = _all_bitstrings(n_obs)
    obs_vec = _hist_to_prob_vec(observed_hist, keys)

    def score(pred_probs: dict[str, float]) -> float:
        pred_vec = _prob_dict_to_vec(pred_probs, keys)
        if metric == "tvd":
            return total_variation_distance(obs_vec, pred_vec)
        if metric == "kl":
            return kl_divergence(obs_vec, pred_vec)
        raise ValueError("metric must be 'tvd' or 'kl'")

    best_p01 = None
    best_p10 = None
    best_score = float("inf")

    samples: list[dict[str, float]] = []
    for p01 in grid:
        for p10 in grid:
            pred = apply_readout_error_probs(
                ideal_probs, p01=float(p01), p10=float(p10)
            )
            s = float(score(pred))
            samples.append({"p01": float(p01), "p10": float(p10), "score": float(s)})
            if s < best_score:
                best_score = s
                best_p01 = float(p01)
                best_p10 = float(p10)

    return {
        "best": {"p01": best_p01, "p10": best_p10, "score": float(best_score)},
        "metric": metric,
        "grid": {
            "min": float(grid.min()),
            "max": float(grid.max()),
            "n": int(grid.size),
        },
        "samples": samples,
    }


def fit_noise_strength_grid(
    observed: dict[str, int],
    grid: np.ndarray,
    simulate_fn: Callable[[float], dict[str, int]],
    metric: str = "tvd",
) -> dict[str, Any]:
    """
    Grid search for best noise strength; return best value + curve + metric scores.
    """
    n = _infer_n_from_hist(observed)
    keys = _all_bitstrings(n)
    obs_vec = _hist_to_prob_vec(observed, keys)

    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be a 1D array with at least 2 points")

    curve: list[dict[str, float]] = []
    best_strength = None
    best_score = float("inf")

    for g in grid:
        pred_hist = simulate_fn(float(g))
        pred_vec = _hist_to_prob_vec(pred_hist, keys)
        if metric == "tvd":
            s = total_variation_distance(obs_vec, pred_vec)
        elif metric == "kl":
            s = kl_divergence(obs_vec, pred_vec)
        else:
            raise ValueError("metric must be 'tvd' or 'kl'")

        curve.append({"strength": float(g), "score": float(s)})
        if s < best_score:
            best_score = float(s)
            best_strength = float(g)

    if best_strength is None:
        raise RuntimeError("internal error: best_strength was not set")

    return {
        "metric": metric,
        "best_strength": float(best_strength),
        "best_score": float(best_score),
        "curve": curve,
    }
