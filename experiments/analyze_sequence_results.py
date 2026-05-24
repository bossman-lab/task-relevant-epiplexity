"""Analyze the sequence MLP transfer experiment and emit SVG figures."""

from __future__ import annotations

import csv
import html
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

RUNS_CSV = RESULTS / "sequence_mlp_transfer_runs.csv"

COLORS = {
    "random": "#7a8794",
    "irrelevant_rule": "#d95f02",
    "relevant_rule": "#1b9e77",
    "mixed_rule": "#7570b3",
}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: pd.Series, y: pd.Series) -> float:
    return pearson(x.rank(method="average").to_numpy(), y.rank(method="average").to_numpy())


def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def project_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def svg_sequence_bars(summary: pd.DataFrame, path: Path) -> None:
    width, height = 920, 410
    margin = 58
    panel_gap = 58
    panel_w = (width - 2 * margin - panel_gap) / 2
    panel_h = height - 2 * margin
    panels = [
        ("learning_progress_proxy", "Learning progress proxy", 0.0, max(1.7, float(summary["learning_progress_proxy"].max()) * 1.15)),
        ("transfer_accuracy", "Transfer accuracy", 0.45, 0.68),
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#263238}.title{font-size:15px;font-weight:700}.axis{stroke:#455a64;stroke-width:1}.grid{stroke:#d9e1e8;stroke-width:1}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    kinds = list(COLORS)
    bar_w = 54
    for pi, (metric, title, y_lo, y_hi) in enumerate(panels):
        px = margin + pi * (panel_w + panel_gap)
        py = margin
        lines.append(f'<text x="{px}" y="{py - 24}" class="title">{html.escape(title)}</text>')
        lines.append(f'<line x1="{px}" y1="{py + panel_h}" x2="{px + panel_w}" y2="{py + panel_h}" class="axis"/>')
        lines.append(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + panel_h}" class="axis"/>')
        for tick in np.linspace(y_lo, y_hi, 5):
            y = scale(tick, y_lo, y_hi, py + panel_h, py)
            lines.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px + panel_w}" y2="{y:.1f}" class="grid"/>')
            lines.append(f'<text x="{px - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:.2f}</text>')
        for i, kind in enumerate(kinds):
            row = summary[summary["kind"] == kind].iloc[0]
            value = float(row[metric])
            x = px + 28 + i * (bar_w + 18)
            y = scale(value, y_lo, y_hi, py + panel_h, py)
            h = py + panel_h - y
            color = COLORS[kind]
            lines.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" fill-opacity="0.82"/>')
            lines.append(f'<text x="{x + bar_w / 2}" y="{y - 6:.1f}" text-anchor="middle">{value:.3f}</text>')
            label = kind.replace("_", " ")
            lines.append(f'<text x="{x + bar_w / 2}" y="{py + panel_h + 18}" text-anchor="middle" transform="rotate(28 {x + bar_w / 2} {py + panel_h + 18})">{html.escape(label)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = pd.read_csv(RUNS_CSV)
    summary = (
        df.groupby("kind", as_index=False)
        .agg(
            initial_val_loss=("initial_val_loss", "mean"),
            final_val_loss=("final_val_loss", "mean"),
            learning_progress_proxy=("learning_progress_proxy", "mean"),
            loss_reduction=("loss_reduction", "mean"),
            transfer_accuracy=("transfer_accuracy", "mean"),
            transfer_accuracy_std=("transfer_accuracy", "std"),
        )
        .sort_values("kind")
    )
    summary.to_csv(RESULTS / "sequence_mlp_transfer_group_means.csv", index=False)

    metrics = ["final_val_loss", "learning_progress_proxy", "loss_reduction"]
    corr_rows = []
    for metric in metrics:
        corr_rows.append(
            {
                "metric": metric,
                "pearson_with_transfer": f"{pearson(df[metric].to_numpy(), df['transfer_accuracy'].to_numpy()):.4f}",
                "spearman_with_transfer": f"{spearman(df[metric], df['transfer_accuracy']):.4f}",
            }
        )
    write_csv(RESULTS / "sequence_mlp_transfer_correlations.csv", corr_rows)

    def mean_value(kind: str, col: str) -> float:
        return float(summary[summary["kind"] == kind].iloc[0][col])

    lp_gap = mean_value("irrelevant_rule", "learning_progress_proxy") - mean_value("relevant_rule", "learning_progress_proxy")
    acc_gap = mean_value("relevant_rule", "transfer_accuracy") - mean_value("irrelevant_rule", "transfer_accuracy")
    mixed_acc = mean_value("mixed_rule", "transfer_accuracy")

    md = [
        "# Sequence MLP Analysis",
        "",
        "Generated by `experiments/analyze_sequence_results.py`.",
        "",
        "## Metric Correlations",
        "",
        "| metric | Pearson with transfer | Spearman with transfer |",
        "|---|---:|---:|",
    ]
    for row in corr_rows:
        md.append(f"| {row['metric']} | {row['pearson_with_transfer']} | {row['spearman_with_transfer']} |")
    md.extend(
        [
            "",
            "## Key Contrasts",
            "",
            f"- `irrelevant_rule` has `{lp_gap:.4f}` more learning-progress proxy than `relevant_rule`, but `relevant_rule` has `{acc_gap:.4f}` higher transfer accuracy.",
            f"- `mixed_rule` transfer accuracy is `{mixed_acc:.4f}`, suggesting that mixed structure can help when some target-relevant signal is present.",
            "- This experiment is weaker than the PCA sanity check but more language-model-like because it uses next-token training and a frozen hidden representation.",
            "",
            "## Figures",
            "",
            "- `experiments/figures/sequence_mlp_bars.svg`",
        ]
    )
    (RESULTS / "sequence_mlp_transfer_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    svg_sequence_bars(summary, FIGURES / "sequence_mlp_bars.svg")
    print(project_path(RESULTS / "sequence_mlp_transfer_analysis.md"))
    print(project_path(FIGURES / "sequence_mlp_bars.svg"))


if __name__ == "__main__":
    main()
