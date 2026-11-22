"""Zip the report folder for Overleaf (main.tex, neurips_2026.sty, figures/*.pdf) into Downloads."""
import zipfile
from pathlib import Path

R = Path.home() / "work/cell-signalling/report"
out = Path("/mnt/c/Users/andre/Downloads/cell-signalling-report-overleaf.zip")
files = [R / "main.tex", R / "neurips_2026.sty", *sorted((R / "figures").glob("*.pdf"))]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(R).as_posix())
print(out, [f.relative_to(R).as_posix() for f in files])
tex = (R / "main.tex").read_text()
print("TEAM prompts left to fill:", tex.count("\\TEAM{") - 1)
