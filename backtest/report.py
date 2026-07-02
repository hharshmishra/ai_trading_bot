"""Backtest report writer: report.json (machine, feeds the HTML deck) +
report.md (human tables)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _fmt(x, pc=False) -> str:
    if x is None:
        return "—"
    if pc:
        return f"{100 * x:.1f}%"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def write_report(run_dir: str, meta: Dict[str, Any], summary: Dict[str, Any],
                 funnel: Dict[str, int], comparison: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(run_dir, exist_ok=True)
    payload = {"meta": meta, "summary": summary, "funnel": funnel}
    if comparison:
        payload["comparison"] = comparison
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [f"# Backtest report — {meta.get('label', 'run')}", ""]
    lines.append(f"- pairs: {len(meta.get('pairs', []))} | tfs: {meta.get('tfs')} | "
                 f"range: {meta.get('start')} → {meta.get('end') or 'now'} | gate: {meta.get('gate')}")
    lines.append(f"- total emissions: {summary.get('total_emissions')}")
    lines.append("- caveat: confidence-gate path uses indicator-only confidence "
                 "(news/research not backtestable); NWE/trend paths exact.")
    lines.append("")
    lines.append("## Per-group metrics (tf | regime | trigger)")
    rows = []
    for key, g in summary.get("groups", {}).items():
        ci = g.get("tb_precision_ci")
        rows.append([
            key, g["n"], g["tp"], g["sl"], g["timeout"],
            _fmt(g.get("tb_precision"), pc=True),
            f"[{_fmt(ci[0], pc=True)}, {_fmt(ci[1], pc=True)}]" if ci else "—",
            _fmt(g.get("expectancy_r")),
            _fmt(g.get("fixed_hit_rate"), pc=True),
        ])
    lines.append(_md_table(
        ["group", "n", "tp", "sl", "t/o", "TB precision", "95% CI", "expectancy R", "fixed hit"],
        rows))
    lines.append("")
    lines.append("## Gate funnel")
    lines.append(_md_table(["outcome", "bars"],
                           sorted(funnel.items(), key=lambda kv: -kv[1])))
    if comparison:
        lines.append("")
        lines.append("## vs baseline")
        crows = []
        for key, c in comparison.get("groups", {}).items():
            if "tb_precision_delta" in c:
                crows.append([key, _fmt(c["tb_precision_delta"], pc=True),
                              _fmt(c.get("expectancy_r_delta")),
                              c.get("n_delta"), "yes" if c.get("significant_95") else "no"])
        lines.append(_md_table(["group", "ΔTB precision", "Δexpectancy", "Δn", "sig@95%"], crows))

    md_path = os.path.join(run_dir, "report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return md_path
