$ErrorActionPreference = "Stop"

$Python = $env:PYTHON
if ([string]::IsNullOrWhiteSpace($Python)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw "Python was not found. Install Python with numpy and pandas, then rerun this script."
    }
    $Python = $PythonCommand.Source
}

& $Python experiments\task_relevant_epiplexity_pca.py
& $Python experiments\analyze_pca_results.py
& $Python experiments\sequence_mlp_transfer.py
& $Python experiments\analyze_sequence_results.py
& $Python experiments\export_paper_figures.py

Write-Host "Generated experiment outputs under experiments/results and experiments/figures."
