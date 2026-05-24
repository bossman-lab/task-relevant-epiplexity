"""Small sequence-model experiment for task-relevant epiplexity.

This is a controlled next-token prediction experiment using a one-hidden-layer
MLP written in NumPy. It is deliberately small enough to run on a laptop while
preserving several ingredients that the PCA sanity check lacks:

- autoregressive next-token training;
- a learning-curve based structure proxy;
- frozen representation transfer to a downstream target task;
- source distributions with learnable but target-irrelevant structure.

The experiment should not be read as a claim about frontier LLMs. It is a
second, more language-model-like controlled test for the paper draft.
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
    seq_len: int = 26
    context: int = 6
    n_source_train: int = 2200
    n_source_val: int = 600
    n_target_train: int = 8
    n_target_test: int = 1600
    hidden: int = 10
    epochs: int = 20
    batch_size: int = 512
    repeats: int = 18
    lr: float = 0.055
    ridge_lambda: float = 0.35
    noise: float = 0.04


SOURCE_KINDS = [
    "random",
    "irrelevant_rule",
    "relevant_rule",
    "mixed_rule",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def bce_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    # Stable binary cross entropy.
    return float(np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))))


def generate_sequences(kind: str, n: int, cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate binary sequences and optional target labels.

    Target labels are meaningful for the target/relevant rule family. The label
    c controls a recurrence that can be inferred from local transitions:

        x_t = x_{t-1} xor x_{t-2} xor x_{t-3} xor c

    Irrelevant source data uses a different recurrence that is learnable for
    next-token prediction but not aligned with this target label.
    """
    x = rng.integers(0, 2, size=(n, cfg.seq_len), dtype=np.int8)
    c = rng.integers(0, 2, size=n, dtype=np.int8)
    d = rng.integers(0, 2, size=n, dtype=np.int8)

    if kind == "random":
        return x, c.astype(np.float64)

    for t in range(4, cfg.seq_len):
        if kind == "relevant_rule":
            nxt = x[:, t - 1] ^ x[:, t - 2] ^ x[:, t - 3] ^ c
        elif kind == "irrelevant_rule":
            nxt = x[:, t - 1] ^ x[:, t - 3] ^ x[:, t - 4] ^ d
        elif kind == "mixed_rule":
            use_relevant = (t % 3) == 0
            rel = x[:, t - 1] ^ x[:, t - 2] ^ x[:, t - 3] ^ c
            irr = x[:, t - 1] ^ x[:, t - 3] ^ x[:, t - 4] ^ d
            nxt = np.where(use_relevant, rel, irr)
        else:
            raise ValueError(f"unknown kind: {kind}")
        flips = rng.random(n) < cfg.noise
        x[:, t] = np.where(flips, 1 - nxt, nxt)

    return x, c.astype(np.float64)


def windows_from_sequences(seqs: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for t in range(cfg.context, cfg.seq_len):
        xs.append(seqs[:, t - cfg.context : t])
        ys.append(seqs[:, t])
    x = np.concatenate(xs, axis=0).astype(np.float64)
    y = np.concatenate(ys, axis=0).astype(np.float64)
    return x, y


def init_model(cfg: Config, rng: np.random.Generator) -> dict[str, np.ndarray]:
    scale = 1.0 / np.sqrt(cfg.context)
    return {
        "w1": rng.normal(0.0, scale, size=(cfg.context, cfg.hidden)),
        "b1": np.zeros(cfg.hidden),
        "w2": rng.normal(0.0, 1.0 / np.sqrt(cfg.hidden), size=(cfg.hidden,)),
        "b2": np.zeros(()),
    }


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.tanh(x @ params["w1"] + params["b1"])
    logits = h @ params["w2"] + params["b2"]
    return h, logits


def train_next_token_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], list[float]]:
    params = init_model(cfg, rng)
    n = len(train_x)
    val_losses: list[float] = []

    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            xb = train_x[idx]
            yb = train_y[idx]
            h, logits = forward(params, xb)
            p = sigmoid(logits)
            dz = (p - yb) / len(xb)

            grad_w2 = h.T @ dz
            grad_b2 = dz.sum()
            dh = dz[:, None] * params["w2"][None, :]
            da = dh * (1.0 - h * h)
            grad_w1 = xb.T @ da
            grad_b1 = da.sum(axis=0)

            params["w2"] -= cfg.lr * grad_w2
            params["b2"] -= cfg.lr * grad_b2
            params["w1"] -= cfg.lr * grad_w1
            params["b1"] -= cfg.lr * grad_b1

        _, val_logits = forward(params, val_x)
        val_losses.append(bce_from_logits(val_logits, val_y))

    return params, val_losses


def sequence_features(params: dict[str, np.ndarray], seqs: np.ndarray, cfg: Config) -> np.ndarray:
    """Average hidden activations over all contexts in each sequence."""
    feats = []
    for t in range(cfg.context, cfg.seq_len):
        x = seqs[:, t - cfg.context : t].astype(np.float64)
        h, _ = forward(params, x)
        feats.append(h)
    stacked = np.stack(feats, axis=1)
    return np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)], axis=1)


def fit_ridge_classifier(z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    y_signed = 2.0 * y - 1.0
    z_aug = np.column_stack([z, np.ones(len(z))])
    penalty = lam * np.eye(z_aug.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(z_aug.T @ z_aug + penalty, z_aug.T @ y_signed)


def ridge_accuracy(z: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    z_aug = np.column_stack([z, np.ones(len(z))])
    pred = (z_aug @ beta >= 0.0).astype(np.float64)
    return float(np.mean(pred == y))


def learning_progress_proxy(losses: list[float]) -> float:
    final = losses[-1]
    return float(sum(max(loss - final, 0.0) for loss in losses))


def one_run(kind: str, repeat: int, cfg: Config) -> dict[str, float | int | str]:
    rng = np.random.default_rng(40_000 + repeat * 97 + SOURCE_KINDS.index(kind))
    train_seqs, _ = generate_sequences(kind, cfg.n_source_train, cfg, rng)
    val_seqs, _ = generate_sequences(kind, cfg.n_source_val, cfg, rng)
    train_x, train_y = windows_from_sequences(train_seqs, cfg)
    val_x, val_y = windows_from_sequences(val_seqs, cfg)

    params, losses = train_next_token_mlp(train_x, train_y, val_x, val_y, cfg, rng)

    target_train, target_train_y = generate_sequences("relevant_rule", cfg.n_target_train, cfg, rng)
    target_test, target_test_y = generate_sequences("relevant_rule", cfg.n_target_test, cfg, rng)
    z_train = sequence_features(params, target_train, cfg)
    z_test = sequence_features(params, target_test, cfg)
    beta = fit_ridge_classifier(z_train, target_train_y, cfg.ridge_lambda)
    transfer_acc = ridge_accuracy(z_test, target_test_y, beta)

    initial_loss = losses[0]
    final_loss = losses[-1]
    lp = learning_progress_proxy(losses)

    return {
        "kind": kind,
        "repeat": repeat,
        "initial_val_loss": initial_loss,
        "final_val_loss": final_loss,
        "learning_progress_proxy": lp,
        "loss_reduction": initial_loss - final_loss,
        "transfer_accuracy": transfer_acc,
    }


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    out = []
    for kind in SOURCE_KINDS:
        group = [r for r in rows if r["kind"] == kind]
        entry = {"kind": kind}
        for field in [
            "initial_val_loss",
            "final_val_loss",
            "learning_progress_proxy",
            "loss_reduction",
            "transfer_accuracy",
        ]:
            vals = np.array([float(r[field]) for r in group])
            entry[field] = f"{vals.mean():.4f} +/- {vals.std(ddof=1):.4f}"
        out.append(entry)
    return out


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, str]], cfg: Config) -> None:
    lines = [
        "# Sequence MLP Transfer Experiment",
        "",
        "A one-hidden-layer MLP is trained for next-token prediction on synthetic",
        "binary sequences. The frozen hidden representation is then used for few-shot",
        "classification of the target recurrence family.",
        "",
        "Setup:",
        "",
        f"- sequence length: `{cfg.seq_len}`",
        f"- context length: `{cfg.context}`",
        f"- hidden units: `{cfg.hidden}`",
        f"- source train sequences: `{cfg.n_source_train}`",
        f"- target few-shot labels: `{cfg.n_target_train}`",
        f"- repeats: `{cfg.repeats}`",
        "",
        "| source | initial val loss | final val loss | learning progress proxy | loss reduction | transfer accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {kind} | {initial_val_loss} | {final_val_loss} | {learning_progress_proxy} | "
            "{loss_reduction} | {transfer_accuracy} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Reading:",
            "",
            "- `irrelevant_rule` is learnable for next-token prediction but does not match",
            "  the target recurrence label.",
            "- `relevant_rule` is aligned with the downstream target and should transfer",
            "  better when the learned hidden representation captures the recurrence.",
            "- `mixed_rule` contains both structures and tests whether a small model can",
            "  preserve task-relevant features while also fitting irrelevant structure.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = Config()
    rows: list[dict[str, float | int | str]] = []
    for repeat in range(cfg.repeats):
        for kind in SOURCE_KINDS:
            rows.append(one_run(kind, repeat, cfg))

    summary = summarize(rows)
    write_csv(RESULTS / "sequence_mlp_transfer_runs.csv", rows)
    write_markdown(RESULTS / "sequence_mlp_transfer_summary.md", summary, cfg)

    print("source,initial_val_loss,final_val_loss,learning_progress,loss_reduction,transfer_accuracy")
    for row in summary:
        print(
            "{kind},{initial_val_loss},{final_val_loss},{learning_progress_proxy},"
            "{loss_reduction},{transfer_accuracy}".format(**row)
        )


if __name__ == "__main__":
    main()
