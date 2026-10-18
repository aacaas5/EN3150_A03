"""Execute the submission notebook in default quick-review mode, replacing it only on success."""
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "FeatureFour_A03_EN3150.ipynb"
TEMP = ROOT / "tmp" / "FeatureFour_A03_EN3150.executed.ipynb"


def main():
    notebook = nbformat.read(SOURCE, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    executed = client.execute()
    nbformat.validate(executed)
    TEMP.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, TEMP)
    os.replace(TEMP, SOURCE)
    print(f"PASS: executed {len(executed.cells)} cells in quick-review mode")


if __name__ == "__main__":
    main()
