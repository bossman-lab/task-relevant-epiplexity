$ErrorActionPreference = "Stop"

$UserMiKTeXPath = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
if (Test-Path $UserMiKTeXPath) {
    $env:Path = "$UserMiKTeXPath;$env:Path"
}

$PdfLatex = Get-Command pdflatex -ErrorAction SilentlyContinue
$Bibtex = Get-Command bibtex -ErrorAction SilentlyContinue

if (-not $PdfLatex) {
    Write-Host "pdflatex is not installed or not on PATH. LaTeX source is available at paper/main.tex."
    exit 1
}

Push-Location $PSScriptRoot
try {
    & $PdfLatex.Source --enable-installer -interaction=nonstopmode main.tex
    if ($Bibtex) {
        & $Bibtex.Source main
        & $PdfLatex.Source --enable-installer -interaction=nonstopmode main.tex
        & $PdfLatex.Source --enable-installer -interaction=nonstopmode main.tex
    } else {
        Write-Host "bibtex is not installed or not on PATH; references may not resolve."
    }
} finally {
    Pop-Location
}
