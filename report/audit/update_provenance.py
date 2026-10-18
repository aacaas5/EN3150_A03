"""Refresh hashes only after final evidence and report validation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = []
    for folder, patterns in (
        (ROOT / "data/splits", ("*.json",)),
        (ROOT / "results/raw", ("*.json", "*.csv")),
        (ROOT / "results/tables", ("*.csv",)),
        (ROOT / "report/tables", ("*.tex",)),
        (ROOT / "report/sections", ("*.tex",)),
        (ROOT / "report/figures", ("*.pdf",)),
        (ROOT / "src", ("*.py",)),
        (ROOT / "scripts", ("*.py",)),
        (ROOT / "report/audit", ("*.py", "*.csv", "*.json")),
    ):
        for pattern in patterns:
            paths.extend(folder.rglob(pattern))
    paths.extend((ROOT / name) for name in (
        "config.py", "README.md", "requirements.txt", "RUN_LOG.md", "TODO.md",
        "FeatureFour_A03_EN3150.ipynb", "report/main.tex", "report/references.bib",
    ))
    provenance = {
        path.relative_to(ROOT).as_posix(): digest(path)
        for path in sorted(set(paths)) if path.exists()
    }
    output = ROOT / "report/result_provenance.json"
    output.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(provenance)} current evidence hashes.")


if __name__ == "__main__":
    main()
