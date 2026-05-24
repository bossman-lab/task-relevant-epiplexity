# Experiments

## Synthetic PCA Transfer

`task_relevant_epiplexity_pca.py` builds a transparent observer where structure
is represented by PCA components learned from source data.

The target label depends only on the last four coordinates. Source distributions
vary in whether their strongest structure is target-relevant:

- `random`: no strong structure.
- `irrelevant_high`: high-variance structure in target-irrelevant coordinates.
- `relevant_mid`: medium-variance structure in target-relevant coordinates.
- `mixed_budget_limited`: both irrelevant and relevant structure, so a small PCA
  budget can miss the useful part.
- `relevant_high`: high-variance target-relevant structure.

Run:

```powershell
python experiments\task_relevant_epiplexity_pca.py
python experiments\analyze_pca_results.py
```

Generated files:

- `results/task_relevant_epiplexity_pca_runs.csv`: per-run metrics.
- `results/task_relevant_epiplexity_pca_summary.md`: group means.
- `results/task_relevant_epiplexity_pca_group_means.csv`: machine-readable
  group means.
- `results/task_relevant_epiplexity_pca_correlations.csv`: baseline metric
  correlations with transfer accuracy.
- `results/task_relevant_epiplexity_pca_analysis.md`: correlations and key
  contrasts.
- `figures/pca_metric_scatter.svg`: total structure and TRE proxy versus
  transfer accuracy.
- `figures/pca_budget_effect.svg`: observer budget effect for the mixed source.
- `figures/framework_diagram.svg`: conceptual diagram for the paper.

## Interpretation

This is a sanity check for the definitions. It does not establish claims about
large language models. Its purpose is to show that total structure and
task-relevant structure can diverge, and that observer budget changes which
structure becomes visible.

## Sequence MLP Transfer

`sequence_mlp_transfer.py` trains a one-hidden-layer MLP for next-token
prediction on synthetic binary sequences, then freezes the hidden representation
and uses a ridge classifier with a few labels for the target recurrence task.

Run:

```powershell
python experiments\sequence_mlp_transfer.py
python experiments\analyze_sequence_results.py
```

Generated files:

- `results/sequence_mlp_transfer_runs.csv`: per-run metrics.
- `results/sequence_mlp_transfer_summary.md`: group means.
- `results/sequence_mlp_transfer_group_means.csv`: machine-readable group
  means.
- `results/sequence_mlp_transfer_correlations.csv`: correlations between
  learning metrics and transfer accuracy.
- `results/sequence_mlp_transfer_analysis.md`: key contrasts.
- `figures/sequence_mlp_bars.svg`: learning progress and transfer bars.

This experiment is weaker than the PCA check but closer to language modeling:
it uses autoregressive next-token training, a learning-curve proxy, and frozen
representations for transfer.
