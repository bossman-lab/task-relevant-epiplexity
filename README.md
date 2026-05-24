# Task-Relevant Epiplexity Draft

This workspace contains a research draft extending epiplexity from task-agnostic
structural information to task-relevant, observer-dependent transfer.

## Files

- `task_relevant_epiplexity_paper_zh.md` — Chinese paper draft.
- `paper/task_relevant_epiplexity_en.md` — English workshop/arXiv-style draft.
- `paper/main.tex` / `paper/references.bib` — LaTeX manuscript source.
- `paper/main.pdf` — compiled LaTeX paper.
- `paper/figures/` — PDF figures generated from experiment outputs.
- `2601.03220.pdf` / `2601.03220.txt` — optional local-only copies of the
  source epiplexity paper used for close reading. These are ignored by git;
  cite or download the paper from arXiv instead of republishing a third-party
  full text.
- `experiments/task_relevant_epiplexity_pca.py` — synthetic PCA transfer
  experiment.
- `experiments/analyze_pca_results.py` — correlation analysis and SVG figure
  generation.
- `experiments/sequence_mlp_transfer.py` — small next-token MLP transfer
  experiment.
- `experiments/analyze_sequence_results.py` — analysis and SVG figure
  generation for the sequence experiment.
- `experiments/results/` — generated CSV and Markdown results.
- `experiments/figures/` — generated SVG figures.
- `run_all_experiments.ps1` — one-command reproduction helper.

## Reproduce

Run the full reproducibility helper:

```powershell
.\run_all_experiments.ps1
```

The helper runs the experiment, analysis, and figure-export scripts in order.
It uses the first `python` executable on `PATH`. To use a specific interpreter,
set `$env:PYTHON` before running the helper.

Or run the same steps manually with any Python environment that has `numpy` and
`pandas`:

```powershell
python experiments\task_relevant_epiplexity_pca.py
python experiments\analyze_pca_results.py
python experiments\sequence_mlp_transfer.py
python experiments\analyze_sequence_results.py
python experiments\export_paper_figures.py
```

The main generated outputs are:

- `experiments/results/task_relevant_epiplexity_pca_summary.md`
- `experiments/results/task_relevant_epiplexity_pca_group_means.csv`
- `experiments/results/task_relevant_epiplexity_pca_correlations.csv`
- `experiments/results/task_relevant_epiplexity_pca_analysis.md`
- `experiments/figures/pca_metric_scatter.svg`
- `experiments/figures/pca_budget_effect.svg`
- `experiments/figures/framework_diagram.svg`
- `experiments/results/sequence_mlp_transfer_summary.md`
- `experiments/results/sequence_mlp_transfer_analysis.md`
- `experiments/figures/sequence_mlp_bars.svg`
- `paper/figures/*.pdf`

To compile the paper with the installed user-profile MiKTeX:

```powershell
.\paper\build.ps1
```

You can also validate manuscript references and figure paths without rebuilding
the PDF:

```powershell
python paper\validate_manuscript.py
```

## Current Evidence

The current experiment is a controlled toy setting, not evidence about frontier
language models. It shows a minimal failure mode for the strong claim that more
structure always means better transfer:

- `irrelevant_high` has high total structure but near-chance transfer.
- `relevant_mid` has less total structure but strong transfer because it aligns
  with the target.
- `mixed_budget_limited` becomes useful only when the observer representation
  budget grows enough to include target-relevant structure.
- The sequence MLP experiment is a second, weaker but more language-model-like
  check: next-token learning progress alone is not predictive of target
  transfer in the controlled setup.

## Next Work

- Add a stronger sequence-model experiment using a tiny Transformer or recurrent
  observer.
- Compare prequential epiplexity estimates against held-out loss, loss slope,
  target alignment, and downstream transfer on less synthetic tasks.
- Tighten venue formatting, theorem statements, and external baseline
  comparisons before submission.
