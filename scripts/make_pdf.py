#!/usr/bin/env python
"""Render Draft_v2.ipynb -> Draft_v2.pdf.

Primary path : nbconvert --to html  ->  headless Chrome --print-to-pdf
Fallback     : nbconvert --to webpdf (needs `pip install nbconvert[webpdf]`
               and `playwright install chromium`)

Run from the repo root so that figures/ resolves.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Draft_v2.ipynb"
HTML = ROOT / "Draft_v2.html"
PDF = ROOT / "Draft_v2.pdf"

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

    # 1. notebook -> self-contained HTML (embeds figures/ as data URIs)
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "html",
         "--embed-images", "--no-input", "--output", HTML.name, str(NB)],
        cwd=ROOT, check=True,
    )
    print(f"wrote {HTML}")

    # 2. HTML -> PDF
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
         "--output", PDF.stem, str(NB)],
        cwd=ROOT, check=True,
    )
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()