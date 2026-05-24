"""Lightweight manuscript validation without requiring a LaTeX installation."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEX = ROOT / "main.tex"
BIB = ROOT / "references.bib"


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")

    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)", bib))
    cite_keys: set[str] = set()
    for body in re.findall(r"\\cite[t|p|alp|author|year|yearpar]*\{([^}]+)\}", tex):
        cite_keys.update(k.strip() for k in body.split(",") if k.strip())

    missing_cites = sorted(cite_keys - bib_keys)
    unused_bib = sorted(bib_keys - cite_keys)

    figure_paths = re.findall(r"\\includegraphics(?:\[[^]]+\])?\{([^}]+)\}", tex)
    missing_figures = [p for p in figure_paths if not (ROOT / p).exists()]

    print(f"cite keys in tex: {len(cite_keys)}")
    print(f"bib entries: {len(bib_keys)}")
    print(f"figures in tex: {len(figure_paths)}")

    if missing_cites:
        print("missing citation keys:")
        for key in missing_cites:
            print(f"  {key}")
    if missing_figures:
        print("missing figures:")
        for path in missing_figures:
            print(f"  {path}")
    if unused_bib:
        print("unused bib entries:")
        for key in unused_bib:
            print(f"  {key}")

    if missing_cites or missing_figures:
        raise SystemExit(1)
    print("manuscript validation passed")


if __name__ == "__main__":
    main()
