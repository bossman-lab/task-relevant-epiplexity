"""Analyze and visualize the synthetic PCA transfer experiment.

The script intentionally avoids heavyweight plotting dependencies. It reads the
CSV produced by task_relevant_epiplexity_pca.py, computes baseline correlations,
and emits small SVG figures that can be embedded in the paper draft.
"""

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

RUNS_CSV = RESULTS / "task_relevant_epiplexity_pca_runs.csv"

KIND_COLORS = {
    "random": "#7a8794",
    "irrelevant_high": "#d95f02",
    "relevant_mid": "#1b9e77",
    "mixed_budget_limited": "#7570b3",
    "relevant_high": "#2c7fb8",
}

METRICS = [
    "total_structure_proxy",
    "target_alignment",
    "tre_proxy",
]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: pd.Series, y: pd.Series) -> float:
    return pearson(x.rank(method="average").to_numpy(), y.rank(method="average").to_numpy())


def mean_std(series: pd.Series) -> str:
    return f"{series.mean():.4f} +/- {series.std(ddof=1):.4f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def project_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def svg_scatter(df: pd.DataFrame, path: Path) -> None:
    width, height = 920, 420
    margin = 52
    panel_gap = 48
    panel_w = (width - 2 * margin - panel_gap) / 2
    panel_h = height - 2 * margin

    panels = [
        ("total_structure_proxy", "Total structure proxy"),
        ("tre_proxy", "TRE proxy"),
    ]
    y_col = "transfer_accuracy"
    y_lo, y_hi = 0.46, 0.86

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#263238}.title{font-size:15px;font-weight:700}.axis{stroke:#455a64;stroke-width:1}.grid{stroke:#d9e1e8;stroke-width:1}.legend{font-size:11px}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for panel_idx, (x_col, title) in enumerate(panels):
        px = margin + panel_idx * (panel_w + panel_gap)
        py = margin
        x_lo = float(df[x_col].min())
        x_hi = float(df[x_col].max())
        pad = (x_hi - x_lo) * 0.08 or 1.0
        x_lo -= pad
        x_hi += pad

        lines.append(f'<text x="{px}" y="{py - 24}" class="title">{html.escape(title)} vs transfer accuracy</text>')
        lines.append(f'<line x1="{px}" y1="{py + panel_h}" x2="{px + panel_w}" y2="{py + panel_h}" class="axis"/>')
        lines.append(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + panel_h}" class="axis"/>')

        for tick in np.linspace(y_lo, y_hi, 5):
            y = scale(tick, y_lo, y_hi, py + panel_h, py)
            lines.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px + panel_w}" y2="{y:.1f}" class="grid"/>')
            lines.append(f'<text x="{px - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:.2f}</text>')
        for tick in np.linspace(x_lo, x_hi, 4):
            x = scale(tick, x_lo, x_hi, px, px + panel_w)
            lines.append(f'<line x1="{x:.1f}" y1="{py + panel_h}" x2="{x:.1f}" y2="{py + panel_h + 4}" class="axis"/>')
            lines.append(f'<text x="{x:.1f}" y="{py + panel_h + 20}" text-anchor="middle">{tick:.1f}</text>')

        lines.append(f'<text x="{px + panel_w / 2}" y="{height - 8}" text-anchor="middle">{html.escape(title)}</text>')
        if panel_idx == 0:
            lines.append(f'<text x="16" y="{py + panel_h / 2}" transform="rotate(-90 16 {py + panel_h / 2})" text-anchor="middle">Transfer accuracy</text>')

        for _, row in df.iterrows():
            x = scale(float(row[x_col]), x_lo, x_hi, px, px + panel_w)
            y = scale(float(row[y_col]), y_lo, y_hi, py + panel_h, py)
            color = KIND_COLORS[str(row["kind"])]
            radius = 3.2 if int(row["k"]) == 4 else 5.0
            opacity = 0.34 if int(row["k"]) == 4 else 0.22
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" fill-opacity="{opacity}" stroke="{color}" stroke-width="0.4"/>'
            )

    legend_x = width - 198
    legend_y = 18
    lines.append(f'<rect x="{legend_x - 10}" y="{legend_y - 12}" width="188" height="94" rx="6" fill="#f7fafc" stroke="#d9e1e8"/>')
    for i, (kind, color) in enumerate(KIND_COLORS.items()):
        y = legend_y + i * 17
        lines.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 12}" y="{y + 4}" class="legend">{html.escape(kind)}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_budget_effect(summary: pd.DataFrame, path: Path) -> None:
    width, height = 720, 360
    margin = 58
    chart_h = height - 2 * margin
    chart_w = width - 2 * margin
    subset = summary[summary["kind"] == "mixed_budget_limited"].sort_values("k")
    metrics = [
        ("target_alignment", "Target alignment", "#7570b3"),
        ("transfer_accuracy", "Transfer accuracy", "#1b9e77"),
    ]
    max_y = 1.0
    bar_w = 58
    group_gap = 120
    group_xs = [margin + 170, margin + 170 + group_gap]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px;fill:#263238}.title{font-size:16px;font-weight:700}.axis{stroke:#455a64;stroke-width:1}.grid{stroke:#d9e1e8;stroke-width:1}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin}" y="30" class="title">Observer budget effect on mixed source</text>',
        f'<line x1="{margin}" y1="{margin + chart_h}" x2="{margin + chart_w}" y2="{margin + chart_h}" class="axis"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + chart_h}" class="axis"/>',
    ]

    for tick in np.linspace(0, max_y, 6):
        y = scale(tick, 0, max_y, margin + chart_h, margin)
        lines.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{margin + chart_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:.1f}</text>')

    for gi, (_, row) in enumerate(subset.iterrows()):
        base_x = group_xs[gi]
        lines.append(f'<text x="{base_x + 20}" y="{margin + chart_h + 24}" text-anchor="middle">k={int(row["k"])}</text>')
        for mi, (metric, label, color) in enumerate(metrics):
            value = float(row[metric])
            x = base_x + mi * (bar_w + 12)
            y = scale(value, 0, max_y, margin + chart_h, margin)
            h = margin + chart_h - y
            lines.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" fill-opacity="0.82"/>')
            lines.append(f'<text x="{x + bar_w / 2}" y="{y - 6:.1f}" text-anchor="middle">{value:.3f}</text>')

    legend_x = margin + chart_w - 190
    for i, (_, label, color) in enumerate(metrics):
        y = 68 + i * 22
        lines.append(f'<rect x="{legend_x}" y="{y - 12}" width="14" height="14" fill="{color}" fill-opacity="0.82"/>')
        lines.append(f'<text x="{legend_x + 22}" y="{y}">{html.escape(label)}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_framework(path: Path) -> None:
    width, height = 980, 280
    boxes = [
        (40, 92, 150, 72, "Data X", "raw structure + noise"),
        (230, 92, 170, 72, "Observer Ω", "model, tokenizer, compute"),
        (440, 92, 175, 72, "SΩ(X)", "extractable structure"),
        (655, 92, 150, 72, "Task Z", "target relevance"),
        (845, 92, 105, 72, "Transfer", "measured gain"),
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs><marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"8\" refY=\"3\" orient=\"auto\"><path d=\"M0,0 L0,6 L9,3 z\" fill=\"#455a64\"/></marker></defs>",
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263238}.title{font-size:19px;font-weight:700}.box-title{font-size:15px;font-weight:700}.box-sub{font-size:12px}.arrow{stroke:#455a64;stroke-width:1.6;marker-end:url(#arrow)}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="38" class="title">Task-relevant epiplexity separates structure from useful structure</text>',
    ]
    for x, y, w, h, title, sub in boxes:
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="#f7fafc" stroke="#b8c7d3"/>')
        lines.append(f'<text x="{x + w / 2}" y="{y + 30}" text-anchor="middle" class="box-title">{html.escape(title)}</text>')
        lines.append(f'<text x="{x + w / 2}" y="{y + 52}" text-anchor="middle" class="box-sub">{html.escape(sub)}</text>')
    arrow_pairs = [(190, 128, 230, 128), (400, 128, 440, 128), (615, 128, 655, 128), (805, 128, 845, 128)]
    for x1, y1, x2, y2 in arrow_pairs:
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2 - 8}" y2="{y2}" class="arrow"/>')
    lines.append('<text x="526" y="215" text-anchor="middle" font-size="13">TREΩ(X; Z) = transfer gain beyond matched controls</text>')
    lines.append('<text x="526" y="238" text-anchor="middle" font-size="13">Context adds temporary structure: CREΩ(K; Y | Q)</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = pd.read_csv(RUNS_CSV)

    summary = (
        df.groupby(["k", "kind"], as_index=False)
        .agg(
            total_structure_proxy=("total_structure_proxy", "mean"),
            target_alignment=("target_alignment", "mean"),
            tre_proxy=("tre_proxy", "mean"),
            transfer_accuracy=("transfer_accuracy", "mean"),
            transfer_accuracy_std=("transfer_accuracy", "std"),
        )
        .sort_values(["k", "kind"])
    )
    summary.to_csv(RESULTS / "task_relevant_epiplexity_pca_group_means.csv", index=False)

    corr_rows: list[dict[str, object]] = []
    for scope, group in [("all_runs", df), *[(f"k={k}", g) for k, g in df.groupby("k")]]:
        for metric in METRICS:
            corr_rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "pearson_with_transfer": f"{pearson(group[metric].to_numpy(), group['transfer_accuracy'].to_numpy()):.4f}",
                    "spearman_with_transfer": f"{spearman(group[metric], group['transfer_accuracy']):.4f}",
                }
            )
    write_csv(RESULTS / "task_relevant_epiplexity_pca_correlations.csv", corr_rows)

    md = [
        "# PCA Experiment Analysis",
        "",
        "This file is generated by `experiments/analyze_pca_results.py`.",
        "",
        "## Metric Correlations",
        "",
        "| scope | metric | Pearson with transfer | Spearman with transfer |",
        "|---|---|---:|---:|",
    ]
    for row in corr_rows:
        md.append(
            f"| {row['scope']} | {row['metric']} | {row['pearson_with_transfer']} | {row['spearman_with_transfer']} |"
        )
    md.extend(
        [
            "",
            "## Key Contrasts",
            "",
        ]
    )

    def mean_value(kind: str, k: int, col: str) -> float:
        row = summary[(summary["kind"] == kind) & (summary["k"] == k)].iloc[0]
        return float(row[col])

    total_gap = mean_value("irrelevant_high", 4, "total_structure_proxy") - mean_value("relevant_mid", 4, "total_structure_proxy")
    acc_gap = mean_value("relevant_mid", 4, "transfer_accuracy") - mean_value("irrelevant_high", 4, "transfer_accuracy")
    mixed_acc_gain = mean_value("mixed_budget_limited", 12, "transfer_accuracy") - mean_value("mixed_budget_limited", 4, "transfer_accuracy")
    mixed_align_gain = mean_value("mixed_budget_limited", 12, "target_alignment") - mean_value("mixed_budget_limited", 4, "target_alignment")

    md.extend(
        [
            f"- At `k=4`, `irrelevant_high` has `{total_gap:.4f}` more total structure proxy than `relevant_mid`, but `relevant_mid` has `{acc_gap:.4f}` higher transfer accuracy.",
            f"- For `mixed_budget_limited`, increasing observer budget from `k=4` to `k=12` raises target alignment by `{mixed_align_gain:.4f}` and transfer accuracy by `{mixed_acc_gain:.4f}`.",
            "- These contrasts are the intended sanity checks for task-relevant epiplexity and observer specification.",
            "",
            "## Figures",
            "",
            "- `experiments/figures/pca_metric_scatter.svg`",
            "- `experiments/figures/pca_budget_effect.svg`",
            "- `experiments/figures/framework_diagram.svg`",
        ]
    )
    (RESULTS / "task_relevant_epiplexity_pca_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    svg_scatter(df, FIGURES / "pca_metric_scatter.svg")
    svg_budget_effect(summary, FIGURES / "pca_budget_effect.svg")
    svg_framework(FIGURES / "framework_diagram.svg")

    print(project_path(RESULTS / "task_relevant_epiplexity_pca_analysis.md"))
    print(project_path(FIGURES / "pca_metric_scatter.svg"))
    print(project_path(FIGURES / "pca_budget_effect.svg"))
    print(project_path(FIGURES / "framework_diagram.svg"))


if __name__ == "__main__":
    main()
