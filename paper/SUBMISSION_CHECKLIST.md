# Submission Checklist

## Manuscript

- [x] English manuscript draft: `paper/main.tex`
- [x] English Markdown working draft: `paper/task_relevant_epiplexity_en.md`
- [x] Chinese working draft: `../task_relevant_epiplexity_paper_zh.md`
- [x] BibTeX file: `paper/references.bib`
- [x] Reproducibility appendix in `main.tex`
- [x] Limitations section in `main.tex`
- [x] Related work section in `main.tex`

## Figures

- [x] Framework figure: `paper/figures/framework_diagram.pdf`
- [x] PCA scatter figure: `paper/figures/pca_metric_scatter.pdf`
- [x] PCA observer-budget figure: `paper/figures/pca_budget_effect.pdf`
- [x] Sequence MLP figure: `paper/figures/sequence_mlp_bars.pdf`

## Experiments

- [x] PCA observer experiment: `experiments/task_relevant_epiplexity_pca.py`
- [x] PCA analysis: `experiments/analyze_pca_results.py`
- [x] Sequence MLP experiment: `experiments/sequence_mlp_transfer.py`
- [x] Sequence MLP analysis: `experiments/analyze_sequence_results.py`
- [x] Paper figure export: `experiments/export_paper_figures.py`
- [x] One-command reproduction: `run_all_experiments.ps1`

## Validation

- [x] Experiments rerun successfully with `run_all_experiments.ps1`
- [x] Figure PDFs generated from result CSVs
- [x] Citation and figure-path validation passed with `paper/validate_manuscript.py`
- [x] PDF compilation with `paper/build.ps1`

## Note

MiKTeX 25.12 is installed under the user profile, and `paper/build.ps1`
prepends the user MiKTeX binary path before running `pdflatex --enable-installer`.
The current compiled output is `paper/main.pdf`. MiKTeX may still recommend
checking for package updates; this is advisory and did not block compilation.
