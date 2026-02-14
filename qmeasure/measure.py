import numpy as np


def apply_readout_error_probs(
    probs: dict[str, float],
    p01: float,
    p10: float,
) -> dict[str, float]:
    """
    Deterministic expected transform for an independent bit-flip readout model:
      0 -> 1 with prob p01
      1 -> 0 with prob p10

    Input and output are probability dictionaries over equal-length bitstrings.
    """
    if not probs:
        return {}

    p01 = float(p01)
    p10 = float(p10)
    if not (0.0 <= p01 <= 1.0 and 0.0 <= p10 <= 1.0):
        raise ValueError("p01 and p10 must be in [0, 1]")

    keys = list(probs.keys())
    n = len(keys[0])
    if n < 1:
        raise ValueError("bitstrings must be non-empty")

    for k in keys:
        if len(k) != n or any(ch not in ("0", "1") for ch in k):
            raise ValueError(
                "all keys must be equal-length bitstrings containing only 0/1"
            )

    # Enumerate the full bitstring space for stable alignment.
    all_bs = [format(i, f"0{n}b") for i in range(2**n)]
    out: dict[str, float] = {b: 0.0 for b in all_bs}

    for src_bs, ps in probs.items():
        ps = float(ps)
        if ps == 0.0:
            continue
        if ps < 0.0:
            raise ValueError("probabilities must be non-negative")

        for t in all_bs:
            pt = 1.0
            for i in range(n):
                a = src_bs[i]
                b = t[i]
                if a == "0" and b == "0":
                    pt *= 1.0 - p01
                elif a == "0" and b == "1":
                    pt *= p01
                elif a == "1" and b == "1":
                    pt *= 1.0 - p10
                else:  # a == "1" and b == "0"
                    pt *= p10
            out[t] += ps * pt

    total = sum(out.values())
    if total <= 0.0:
        raise ValueError("invalid input distribution (sum <= 0 after transform)")
    for k in list(out.keys()):
        out[k] = out[k] / total

    return out


def measure_projective(
    rho: np.ndarray,
    shots: int,
    seed: int = 42,
    readout_confusion: np.ndarray | None = None,
) -> dict[str, int]:
    """
    Sample bitstrings from rho's diagonal (computational basis).
    If readout_confusion is provided, it must be a 2x2 confusion matrix for independent per-qubit readout:
        [[P(0|0), P(1|0)],
         [P(0|1), P(1|1)]]
    Returns histogram like {"00": 123, "01": 456, ...}
    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square density matrix")

    dim = rho.shape[0]
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError("rho dimension must be a power of 2")

    probs = np.real(np.diag(rho)).clip(0, 1)
    s = probs.sum()
    if s <= 0:
        raise ValueError("Invalid density matrix diagonal (sum <= 0)")
    probs = probs / s

    rng = np.random.default_rng(seed)
    idx = rng.choice(dim, size=shots, p=probs)

    hist: dict[str, int] = {}
    for i in idx:
        b = format(int(i), f"0{n_qubits}b")
        hist[b] = hist.get(b, 0) + 1

    # Optional: apply independent readout confusion to the sampled histogram
    if readout_confusion is not None:
        if readout_confusion.shape != (2, 2):
            raise ValueError("readout_confusion must be 2x2 for MVP")
        # Convert confusion matrix into p01/p10 (independent model)
        # Confusion: rows = true bit (0/1), cols = observed bit (0/1)
        #   P(obs=1|true=0) = p01
        #   P(obs=0|true=1) = p10
        p01 = float(readout_confusion[0, 1])
        p10 = float(readout_confusion[1, 0])
        hist = apply_readout_error(hist, p01=p01, p10=p10, seed=seed + 999)

    return hist


def apply_readout_error(
    histogram: dict[str, int], p01: float, p10: float, seed: int = 42
) -> dict[str, int]:
    """
    Flip bits probabilistically to simulate readout noise (independent per bit):
      0 -> 1 with prob p01
      1 -> 0 with prob p10
    """
    p01 = float(p01)
    p10 = float(p10)
    if not (0.0 <= p01 <= 1.0 and 0.0 <= p10 <= 1.0):
        raise ValueError("p01 and p10 must be in [0, 1]")

    rng = np.random.default_rng(seed)
    out: dict[str, int] = {}

    for bitstr, c in histogram.items():
        if c <= 0:
            continue
        for _ in range(int(c)):
            bits = list(bitstr)
            for j, bj in enumerate(bits):
                r = rng.random()
                if bj == "0":
                    if r < p01:
                        bits[j] = "1"
                else:  # bj == "1"
                    if r < p10:
                        bits[j] = "0"
            bs2 = "".join(bits)
            out[bs2] = out.get(bs2, 0) + 1

    return out
