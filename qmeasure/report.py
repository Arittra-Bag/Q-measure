import csv
import json
from pathlib import Path
from typing import Any, TypeGuard


def _is_table(rows: Any) -> TypeGuard[list[dict[str, Any]]]:
    if not isinstance(rows, list) or not rows:
        return False
    if not all(isinstance(r, dict) for r in rows):
        return False
    keys = set(rows[0].keys())
    return all(set(r.keys()) == keys for r in rows)


def save_results(out_dir: str, payload: dict[str, Any]) -> str:
    """
    Save results.json and optional CSV summaries into out_dir.

    - results.json: always written.
    - results.csv: written if payload["results"] is a list[dict] with consistent keys.
    - histogram.csv: written if payload["histogram"] is a dict[str,int].
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    rows = payload.get("results")
    if _is_table(rows):
        csv_path = out_path / "results.csv"
        fieldnames = sorted(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            dw = csv.DictWriter(f, fieldnames=fieldnames)
            dw.writeheader()
            for r in rows:
                dw.writerow(r)

    hist = payload.get("histogram")
    if isinstance(hist, dict) and all(isinstance(k, str) for k in hist.keys()):
        hcsv = out_path / "histogram.csv"
        with hcsv.open("w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            cw.writerow(["bitstring", "count"])
            for k in sorted(hist.keys()):
                cw.writerow([k, int(hist.get(k, 0))])

    return str(json_path)


def _fmt_noise(noise: Any) -> str:
    if not noise:
        return "none"
    try:
        parts = []
        for ns in noise:
            kind = getattr(ns, "kind", None) or (
                ns.get("kind") if isinstance(ns, dict) else None
            )
            strength = (
                getattr(ns, "strength", None)
                if not isinstance(ns, dict)
                else ns.get("strength")
            )
            targets = (
                getattr(ns, "targets", None)
                if not isinstance(ns, dict)
                else ns.get("targets")
            )
            if targets is None:
                parts.append(f"{kind}({strength})")
            else:
                parts.append(f"{kind}({strength}, targets={targets})")
        return ", ".join(parts)
    except Exception:
        return str(noise)


def make_markdown_report(payload: dict[str, Any]) -> str:
    """
    Create a minimal ASCII-only Markdown report summarizing config, calibration, and uncertainty.
    """
    title = payload.get("experiment")
    if not title:
        cfg = payload.get("config") or {}
        title = cfg.get("name", "Q-measure Report")

    cfg = payload.get("config") or {}
    n_qubits = cfg.get("n_qubits", payload.get("n_qubits"))
    shots = cfg.get("shots", payload.get("shots"))

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    if n_qubits is not None or shots is not None:
        lines.append("## Config")
        if n_qubits is not None:
            lines.append(f"- n_qubits: {n_qubits}")
        if shots is not None:
            lines.append(f"- shots: {shots}")
        noise = cfg.get("noise", payload.get("noise"))
        if noise is not None:
            lines.append(f"- noise: {_fmt_noise(noise)}")
        lines.append("")

    fit = payload.get("fit")
    if isinstance(fit, dict):
        lines.append("## Calibration")
        for k in ["p01", "p10", "score", "tvd", "metric"]:
            if k in fit:
                lines.append(f"- {k}: {fit[k]}")
        # If fit has nested best
        if "best" in fit and isinstance(fit["best"], dict):
            b = fit["best"]
            for k in ["p01", "p10", "score"]:
                if k in b:
                    lines.append(f"- best_{k}: {b[k]}")
        lines.append("")

    mc = payload.get("mc")
    if isinstance(mc, dict):
        lines.append("## Uncertainty (MC)")
        if "n_trajectories" in mc:
            lines.append(f"- n_trajectories: {mc['n_trajectories']}")
        if "alpha" in mc:
            lines.append(f"- alpha: {mc['alpha']} (quantiles at alpha/2 and 1-alpha/2)")
        lines.append("- bands: p_lo_probs / p_hi_probs per bitstring")
        lines.append("")

    if "histogram" in payload and isinstance(payload["histogram"], dict):
        lines.append("## Histogram")
        lines.append("- see histogram.csv")
        lines.append("")

    return "\n".join(lines)
