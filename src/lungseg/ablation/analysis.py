"""Ablation analysis: summaries, violin plots and Wilcoxon tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


def _markdown_table(df: pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def analyze(outputs_dir: Path) -> Path:
    outputs_dir = Path(outputs_dir)
    ablation_dir = outputs_dir / "ablation"
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ablation_dir.glob("*.json"))]
    if not rows:
        raise FileNotFoundError(f"no ablation JSON files found in {ablation_dir}")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["data_fraction", "augment_regime"], as_index=False)
        .agg(
            dice_median=("best_val_dice", "median"),
            dice_q1=("best_val_dice", lambda s: s.quantile(0.25)),
            dice_q3=("best_val_dice", lambda s: s.quantile(0.75)),
            hd95_median=("best_val_hd95", "median"),
            hd95_q1=("best_val_hd95", lambda s: s.quantile(0.25)),
            hd95_q3=("best_val_hd95", lambda s: s.quantile(0.75)),
            n=("best_val_dice", "count"),
        )
        .sort_values(["data_fraction", "augment_regime"])
    )
    summary_path = outputs_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_path = outputs_dir / "ablation_violin.png"
    try:
        import matplotlib.pyplot as plt

        df = df.copy()
        df["cell"] = df["data_fraction"].astype(str) + " / " + df["augment_regime"].astype(str)
        fig, ax = plt.subplots(figsize=(10, 4))
        ordered = sorted(df["cell"].unique())
        ax.violinplot([df.loc[df["cell"] == cell, "best_val_dice"].dropna() for cell in ordered])
        ax.set_xticks(range(1, len(ordered) + 1), ordered, rotation=30, ha="right")
        ax.set_ylabel("Best val Dice")
        ax.set_title("Phase 6 ablation")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except Exception:
        plot_path = Path("")

    wilcoxon_rows = []
    for fraction in sorted(df["data_fraction"].unique()):
        sub = df[df["data_fraction"] == fraction]
        pivot = sub.pivot_table(index="seed", columns="augment_regime", values="best_val_dice")
        if {"none", "standard"}.issubset(pivot.columns) and len(pivot.dropna()) >= 2:
            try:
                stat, p_value = wilcoxon(pivot["none"], pivot["standard"])
                wilcoxon_rows.append((fraction, float(stat), float(p_value)))
            except ValueError:
                wilcoxon_rows.append((fraction, float("nan"), float("nan")))

    report_path = outputs_dir / "REPORT_ABLATION.md"
    lines = [
        "# REPORT_ABLATION",
        "",
        "## Summary",
        "",
        *_markdown_table(summary),
        "",
        "## Wilcoxon paired tests",
        "",
    ]
    if wilcoxon_rows:
        lines.append("| data_fraction | statistic | p_value |")
        lines.append("|---:|---:|---:|")
        for fraction, stat, p_value in wilcoxon_rows:
            lines.append(f"| {fraction:g} | {stat:.4g} | {p_value:.4g} |")
    else:
        lines.append("Not enough matched seeds to run Wilcoxon tests.")
    lines.extend(["", f"- Summary CSV: `{summary_path}`"])
    if str(plot_path):
        lines.append(f"- Violin plot: `{plot_path}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
