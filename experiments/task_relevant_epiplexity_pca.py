"""Synthetic proof-of-concept for task-relevant epiplexity.

This experiment is intentionally small and controlled. It uses a linear PCA
"observer" so the relevant quantities are easy to inspect:

- total_structure_proxy: how much non-isotropic structure the frozen source
  representation captures;
- target_alignment: how much that frozen representation overlaps the target
  task direction;
- tre_proxy: their product, a toy proxy for task-relevant epiplexity;
- transfer_accuracy: few-shot target performance using only the frozen
  representation.

The point is not to model LLMs. The point is to produce a counterexample to the
strong claim "more structure always means better transfer": a high-structure
source can be useless for a target task when the learned structure is unrelated.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    d: int = 32
    n_source: int = 6000
    n_target_train: int = 24
    n_target_test: int = 4000
    repeats: int = 64
    label_noise: float = 0.45
    ridge_lambda: float = 0.15


SOURCE_KINDS = [
    "random",
    "irrelevant_high",
    "relevant_mid",
    "mixed_budget_limited",
    "relevant_high",
]


def target_direction(d: int) -> np.ndarray:
    """A fixed sparse target direction in the last four coordinates."""
    w = np.zeros(d)
    w[-4:] = np.array([0.35, -0.65, 0.50, 0.45])
    return w / np.linalg.norm(w)


def source_variances(kind: str, cfg: Config) -> np.ndarray:
    """Diagonal covariance spectrum for each source distribution."""
    v = np.ones(cfg.d)
    target_dims = np.arange(cfg.d - 4, cfg.d)
    irrelevant_dims = np.arange(0, 8)

    if kind == "random":
        pass
    elif kind == "irrelevant_high":
        v[irrelevant_dims] = 8.0
    elif kind == "relevant_mid":
        v[target_dims] = 3.0
    elif kind == "mixed_budget_limited":
        v[irrelevant_dims] = 8.0
        v[target_dims] = 3.0
    elif kind == "relevant_high":
        v[target_dims] = 8.0
    else:
        raise ValueError(f"unknown source kind: {kind}")
    return v


def sample_source(kind: str, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(source_variances(kind, cfg))
    return rng.normal(size=(cfg.n_source, cfg.d)) * std


def sample_target(
    cfg: Config,
    rng: np.random.Generator,
    w: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, cfg.d))
    score = x @ w + cfg.label_noise * rng.normal(size=n)
    y = np.where(score >= 0.0, 1.0, -1.0)
    return x, y


def fit_pca(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / (len(xc) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return mean, vals[:k], vecs[:, :k]


def fit_ridge_classifier(z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    z_aug = np.column_stack([z, np.ones(len(z))])
    penalty = lam * np.eye(z_aug.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(z_aug.T @ z_aug + penalty, z_aug.T @ y)


def ridge_accuracy(z: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    z_aug = np.column_stack([z, np.ones(len(z))])
    pred = np.where(z_aug @ beta >= 0.0, 1.0, -1.0)
    return float(np.mean(pred == y))


def one_run(
    cfg: Config,
    kind: str,
    k: int,
    seed: int,
    target_train: tuple[np.ndarray, np.ndarray],
    target_test: tuple[np.ndarray, np.ndarray],
    w: np.ndarray,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    source = sample_source(kind, cfg, rng)
    mean, eigvals, basis = fit_pca(source, k)

    avg_var = float(np.trace(np.cov((source - source.mean(axis=0)).T)) / cfg.d)
    total_structure = float(np.maximum(eigvals - avg_var, 0.0).sum())

    # Fraction of the target direction contained in the frozen representation.
    target_alignment = float(np.linalg.norm(basis.T @ w) ** 2)
    tre_proxy = total_structure * target_alignment

    x_train, y_train = target_train
    x_test, y_test = target_test
    z_train = (x_train - mean) @ basis
    z_test = (x_test - mean) @ basis
    beta = fit_ridge_classifier(z_train, y_train, cfg.ridge_lambda)
    acc = ridge_accuracy(z_test, y_test, beta)

    return {
        "kind": kind,
        "k": k,
        "seed": seed,
        "total_structure_proxy": total_structure,
        "target_alignment": target_alignment,
        "tre_proxy": tre_proxy,
        "transfer_accuracy": acc,
    }


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for k in sorted({int(r["k"]) for r in rows}):
        for kind in SOURCE_KINDS:
            group = [r for r in rows if int(r["k"]) == k and r["kind"] == kind]
            if not group:
                continue
            summary = {"k": str(k), "kind": kind}
            for field in [
                "total_structure_proxy",
                "target_alignment",
                "tre_proxy",
                "transfer_accuracy",
            ]:
                vals = np.array([float(r[field]) for r in group])
                summary[field] = f"{vals.mean():.4f} +/- {vals.std(ddof=1):.4f}"
            out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fields = [
        "kind",
        "k",
        "seed",
        "total_structure_proxy",
        "target_alignment",
        "tre_proxy",
        "transfer_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, str]], cfg: Config) -> None:
    lines = [
        "# Synthetic PCA Transfer Experiment",
        "",
        "This is a controlled toy experiment for the paper draft. It uses a PCA observer,",
        "so it should be read as a proof-of-concept for the definitions rather than as",
        "evidence about frontier language models.",
        "",
        "Setup:",
        "",
        f"- input dimension: `{cfg.d}`",
        f"- source samples per run: `{cfg.n_source}`",
        f"- target few-shot labels: `{cfg.n_target_train}`",
        f"- target test samples: `{cfg.n_target_test}`",
        f"- repeats: `{cfg.repeats}`",
        "- target task depends only on the last four coordinates",
        "- `irrelevant_high` has strong source structure in the first eight coordinates",
        "- `relevant_mid` has weaker source structure aligned with the target coordinates",
        "- `mixed_budget_limited` contains both, but a small representation budget spends",
        "  its top components on the irrelevant high-variance structure",
        "",
        "| k | source | total structure proxy | target alignment | TRE proxy | transfer accuracy |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {k} | {kind} | {total_structure_proxy} | {target_alignment} | "
            "{tre_proxy} | {transfer_accuracy} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Reading:",
            "",
            "- `irrelevant_high` has larger total structure than `relevant_mid`, but near-zero",
            "  target alignment and chance-level transfer.",
            "- `relevant_mid` has less total structure, but high task alignment and strong",
            "  few-shot transfer.",
            "- `mixed_budget_limited` shows observer dependence: with `k=4`, the observer",
            "  misses the target-relevant structure; with `k=12`, the same source becomes",
            "  highly transferable because the representation budget can include both",
            "  irrelevant and relevant structure.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = Config()
    w = target_direction(cfg.d)
    rows: list[dict[str, float | int | str]] = []

    for repeat in range(cfg.repeats):
        target_rng = np.random.default_rng(10_000 + repeat)
        target_train = sample_target(cfg, target_rng, w, cfg.n_target_train)
        target_test = sample_target(cfg, target_rng, w, cfg.n_target_test)

        for k in (4, 12):
            for idx, kind in enumerate(SOURCE_KINDS):
                seed = 100_000 + repeat * 100 + k * 10 + idx
                rows.append(one_run(cfg, kind, k, seed, target_train, target_test, w))

    summary = summarize(rows)
    write_csv(RESULTS / "task_relevant_epiplexity_pca_runs.csv", rows)
    write_markdown(RESULTS / "task_relevant_epiplexity_pca_summary.md", summary, cfg)

    # Print a compact table for the command line.
    print("k,source,total_structure,target_alignment,tre_proxy,transfer_accuracy")
    for row in summary:
        print(
            "{k},{kind},{total_structure_proxy},{target_alignment},{tre_proxy},"
            "{transfer_accuracy}".format(**row)
        )


if __name__ == "__main__":
    main()
