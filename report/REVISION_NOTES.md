# Report-only revision

The current revised output is **main_revised.pdf**, compiled from **main.tex**.
The previous main.pdf was locked by another application during compilation,
so a separate output name was used. The earlier PDF is not this revision.

This revision adds explicit figure/table references, captioned architecture
tables, labelled equations, a direct parameter-budget check, clearer metric
definitions, optimizer and transfer-learning interpretation, resource trade-offs,
and an assignment-to-evidence appendix. Experimental CSVs, plots, checkpoints,
training/evaluation code, README and RUN_LOG were not modified.

Numeric tables and LaTeX value macros are derived from the saved results using
`report/audit/build_numeric_assets.py`. Check them without writing files:

```powershell
python report/audit/build_numeric_assets.py --check
python report/audit/verify_report.py
```

Recompile this editorial revision from the project root:

```powershell
Push-Location report
pdflatex -jobname=main_revised -interaction=nonstopmode -halt-on-error main.tex
bibtex main_revised
pdflatex -jobname=main_revised -interaction=nonstopmode -halt-on-error main.tex
pdflatex -jobname=main_revised -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

Do not run the original `scripts.generate_report_assets` solely to compile this
revision: that experiment-era generator would overwrite the revised prose.
The report-only numeric builder preserves the prose and original result files.

The final audit is recorded in `audit/verification.json`; original evidence
fingerprints are in `audit/evidence_hashes_before.json`. Rendering and visual
review artifacts are in `audit/pages/` and `audit/review-*.png`.
The final document has 21 pages. All 115 report checks passed, with no unresolved
LaTeX or BibTeX warnings. The visual review is recorded in `audit/visual_review.md`.

Remaining human input: group number, four member names and four index numbers
in main.tex. Recompile after filling them in.
