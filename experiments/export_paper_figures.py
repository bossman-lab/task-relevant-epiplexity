"""Export paper-ready PDF figures from experiment outputs.

The SVG figures are useful for quick viewing, but LaTeX workflows usually prefer
PDF. This script uses reportlab, which is available in the bundled runtime, to
draw compact PDF versions of the main figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
PAPER_FIGURES = ROOT / "paper" / "figures"
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)


COLORS = {
    "random": colors.HexColor("#7a8794"),
    "irrelevant_high": colors.HexColor("#d95f02"),
    "relevant_mid": colors.HexColor("#1b9e77"),
    "mixed_budget_limited": colors.HexColor("#7570b3"),
    "relevant_high": colors.HexColor("#2c7fb8"),
    "irrelevant_rule": colors.HexColor("#d95f02"),
    "relevant_rule": colors.HexColor("#1b9e77"),
    "mixed_rule": colors.HexColor("#7570b3"),
}


def scale(v: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2.0
    return out_lo + (v - lo) * (out_hi - out_lo) / (hi - lo)


def draw_axes(c: canvas.Canvas, x: float, y: float, w: float, h: float, y_lo: float, y_hi: float) -> None:
    c.setStrokeColor(colors.HexColor("#455a64"))
    c.line(x, y, x + w, y)
    c.line(x, y, x, y + h)
    c.setFont("Helvetica", 7)
    c.setStrokeColor(colors.HexColor("#d9e1e8"))
    for tick in np.linspace(y_lo, y_hi, 5):
        yy = scale(float(tick), y_lo, y_hi, y, y + h)
        c.line(x, yy, x + w, yy)
        c.setFillColor(colors.HexColor("#263238"))
        c.drawRightString(x - 4, yy - 2, f"{tick:.2f}")


def pca_scatter() -> None:
    df = pd.read_csv(RESULTS / "task_relevant_epiplexity_pca_runs.csv")
    out = PAPER_FIGURES / "pca_metric_scatter.pdf"
    c = canvas.Canvas(str(out), pagesize=(640, 300))
    c.setTitle("PCA metric scatter")
    panels = [
        ("total_structure_proxy", "Total structure vs transfer"),
        ("tre_proxy", "TRE proxy vs transfer"),
    ]
    margin = 42
    panel_w = 250
    panel_h = 205
    gap = 55
    y_lo, y_hi = 0.46, 0.86
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, 275, "Total structure can decouple from transfer")
    for i, (col, title) in enumerate(panels):
        x0 = margin + i * (panel_w + gap)
        y0 = 45
        vals = df[col].to_numpy()
        x_lo = float(vals.min() - 0.08 * (vals.max() - vals.min() + 1e-9))
        x_hi = float(vals.max() + 0.08 * (vals.max() - vals.min() + 1e-9))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x0, y0 + panel_h + 18, title)
        draw_axes(c, x0, y0, panel_w, panel_h, y_lo, y_hi)
        c.setFont("Helvetica", 7)
        for tick in np.linspace(x_lo, x_hi, 4):
            xx = scale(float(tick), x_lo, x_hi, x0, x0 + panel_w)
            c.drawCentredString(xx, y0 - 12, f"{tick:.1f}")
        for _, row in df.iterrows():
            xx = scale(float(row[col]), x_lo, x_hi, x0, x0 + panel_w)
            yy = scale(float(row["transfer_accuracy"]), y_lo, y_hi, y0, y0 + panel_h)
            c.setFillColor(COLORS[row["kind"]])
            radius = 2.0 if int(row["k"]) == 4 else 2.8
            c.circle(xx, yy, radius, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#263238"))
        c.drawCentredString(x0 + panel_w / 2, y0 - 25, col.replace("_", " "))
    c.save()


def pca_budget() -> None:
    df = pd.read_csv(RESULTS / "task_relevant_epiplexity_pca_group_means.csv")
    subset = df[df["kind"] == "mixed_budget_limited"].sort_values("k")
    out = PAPER_FIGURES / "pca_budget_effect.pdf"
    c = canvas.Canvas(str(out), pagesize=(430, 260))
    c.setTitle("Observer budget effect")
    x0, y0, w, h = 55, 42, 320, 170
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x0, 232, "Observer budget reveals target-relevant structure")
    draw_axes(c, x0, y0, w, h, 0.0, 1.0)
    bar_w = 34
    group_xs = [145, 255]
    metrics = [
        ("target_alignment", "alignment", colors.HexColor("#7570b3")),
        ("transfer_accuracy", "accuracy", colors.HexColor("#1b9e77")),
    ]
    c.setFont("Helvetica", 8)
    for gi, (_, row) in enumerate(subset.iterrows()):
        base = group_xs[gi]
        c.setFillColor(colors.HexColor("#263238"))
        c.drawCentredString(base + 25, y0 - 16, f"k={int(row['k'])}")
        for mi, (metric, _, color) in enumerate(metrics):
            val = float(row[metric])
            xx = base + mi * 42
            yy = scale(val, 0.0, 1.0, y0, y0 + h)
            c.setFillColor(color)
            c.rect(xx, y0, bar_w, yy - y0, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#263238"))
            c.drawCentredString(xx + bar_w / 2, yy + 4, f"{val:.2f}")
    c.setFillColor(colors.HexColor("#7570b3"))
    c.rect(285, 224, 8, 8, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#263238"))
    c.drawString(298, 224, "target alignment")
    c.setFillColor(colors.HexColor("#1b9e77"))
    c.rect(285, 211, 8, 8, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#263238"))
    c.drawString(298, 211, "transfer accuracy")
    c.save()


def sequence_bars() -> None:
    df = pd.read_csv(RESULTS / "sequence_mlp_transfer_group_means.csv")
    out = PAPER_FIGURES / "sequence_mlp_bars.pdf"
    c = canvas.Canvas(str(out), pagesize=(640, 300))
    c.setTitle("Sequence MLP bars")
    panels = [
        ("learning_progress_proxy", "Learning-progress proxy", 0.0, 1.9),
        ("transfer_accuracy", "Transfer accuracy", 0.45, 0.68),
    ]
    margin = 42
    panel_w = 250
    panel_h = 190
    gap = 55
    kinds = ["random", "irrelevant_rule", "relevant_rule", "mixed_rule"]
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, 275, "Learning progress is not target relevance")
    for i, (metric, title, y_lo, y_hi) in enumerate(panels):
        x0 = margin + i * (panel_w + gap)
        y0 = 50
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x0, y0 + panel_h + 18, title)
        draw_axes(c, x0, y0, panel_w, panel_h, y_lo, y_hi)
        for j, kind in enumerate(kinds):
            row = df[df["kind"] == kind].iloc[0]
            val = float(row[metric])
            xx = x0 + 18 + j * 55
            yy = scale(val, y_lo, y_hi, y0, y0 + panel_h)
            c.setFillColor(COLORS[kind])
            c.rect(xx, y0, 36, yy - y0, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#263238"))
            c.setFont("Helvetica", 7)
            c.drawCentredString(xx + 18, yy + 4, f"{val:.2f}")
            c.saveState()
            c.translate(xx + 18, y0 - 12)
            c.rotate(25)
            c.drawCentredString(0, 0, kind.replace("_", " "))
            c.restoreState()
    c.save()


def framework() -> None:
    out = PAPER_FIGURES / "framework_diagram.pdf"
    c = canvas.Canvas(str(out), pagesize=(640, 210))
    c.setTitle("Framework diagram")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(35, 185, "Task-relevant epiplexity separates structure from useful structure")
    boxes = [
        (35, 95, 80, 42, "Data X", "structure+noise"),
        (145, 95, 95, 42, "Observer", "Omega"),
        (275, 95, 95, 42, "S_Omega(X)", "learned structure"),
        (405, 95, 75, 42, "Task Z", "relevance"),
        (520, 95, 75, 42, "Transfer", "gain"),
    ]
    for x, y, w, h, title, sub in boxes:
        c.setFillColor(colors.HexColor("#f7fafc"))
        c.setStrokeColor(colors.HexColor("#b8c7d3"))
        c.roundRect(x, y, w, h, 5, stroke=1, fill=1)
        c.setFillColor(colors.HexColor("#263238"))
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(x + w / 2, y + 25, title)
        c.setFont("Helvetica", 7)
        c.drawCentredString(x + w / 2, y + 11, sub)
    c.setStrokeColor(colors.HexColor("#455a64"))
    for x1, x2 in [(115, 145), (240, 275), (370, 405), (480, 520)]:
        c.line(x1, 116, x2 - 5, 116)
        c.line(x2 - 10, 121, x2 - 5, 116)
        c.line(x2 - 10, 111, x2 - 5, 116)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#263238"))
    c.drawCentredString(320, 55, "TRE_Omega(X; Z) = target transfer gain beyond matched controls")
    c.drawCentredString(320, 39, "CRE_Omega(K; Y | Q) = inference-time structure supplied by context")
    c.save()


def main() -> None:
    pca_scatter()
    pca_budget()
    sequence_bars()
    framework()
    for path in sorted(PAPER_FIGURES.glob("*.pdf")):
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
