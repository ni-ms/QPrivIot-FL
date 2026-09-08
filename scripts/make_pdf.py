#!/usr/bin/env python
"""Render paper/ipynb/Draft_v2.ipynb -> Draft_v2.pdf.

Draft_v2.ipynb is the single, canonical notebook mirror of the paper (the earlier
Draft.ipynb was superseded and removed).

Pipeline    : nbconvert --execute (populate the outputs)
              -> nbconvert --to html --no-input (hide code, keep the regenerated tables)
              -> headless Chrome --print-to-pdf
Fallback    : nbconvert --to webpdf (needs `pip install nbconvert[webpdf]`
              and `playwright install chromium`)

The execute step is NOT optional: the notebook is stored without outputs, and every
table in section 7 is *generated* by the bottom cell from experiment_results/*.json.
Rendering with --no-input but no execution produces a paper with no numbers in it.

Run from anywhere; paths resolve against the repo root so figures/ and
experiment_results/ are found.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "paper" / "ipynb" / "Draft_v2.ipynb"
EXECUTED = ROOT / "paper" / "ipynb" / "Draft_v2.executed.ipynb"
HTML = ROOT / "paper" / "ipynb" / "Draft_v2.html"
PDF = ROOT / "paper" / "ipynb" / "Draft_v2.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def find_browser():
    for name in ("chrome", "msedge", "chromium"):
        p = shutil.which(name)
        if p:
            return p
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    return None


def main():
    if not NB.exists():
        sys.exit(f"missing {NB}")

    # 1. execute, so the bottom cell regenerates the section-7 tables from the released
    #    grid. Run with cwd=ROOT so its repo-root walk and experiment_results/ lookups
    #    resolve. Output goes to a scratch copy: the committed notebook stays output-free.
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
         "--output", EXECUTED.name, str(NB)],
        cwd=ROOT, check=True,
    )
    print(f"executed -> {EXECUTED}")

    try:
        # 2. executed notebook -> self-contained HTML (embeds figures/ as data URIs)
        subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "html",
             "--embed-images", "--no-input", "--output", HTML.name, str(EXECUTED)],
            cwd=ROOT, check=True,
        )
        print(f"wrote {HTML}")

        # 3. HTML -> PDF
        browser = find_browser()
        if browser:
            subprocess.run(
                [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                 f"--print-to-pdf={PDF}", HTML.as_uri()],
                cwd=ROOT, check=True,
            )
            print(f"wrote {PDF}")
            return

        print("no Chrome/Edge found; falling back to nbconvert webpdf", file=sys.stderr)
        subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "webpdf",
             "--allow-chromium-download", "--no-input",
             "--output", PDF.stem, str(EXECUTED)],
            cwd=ROOT, check=True,
        )
        print(f"wrote {PDF}")
    finally:
        EXECUTED.unlink(missing_ok=True)


if __name__ == "__main__":
    main()