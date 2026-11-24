iSKYLAB mixed-phase paper, working draft v0.3 (17 Nov 2026)

main.tex        Copernicus/ACP manuscript (needs copernicus.cls/.bst from the template or the warm-paper project)
standalone.tex  same text with the article class, for a quick PDF without the Copernicus files
abstract.tex, body.tex, refs.tex   shared text (edit these; both main.tex and standalone.tex read them)
figs/           figures: fig01-03 from analysis/mixed_phase/paper_figs.py, F04-F12 from paper_figs_v3.py

Red [TODO: ...] notes are addressed to named people. Every number is in analysis/mixed_phase/paper/numbers_v3.json
(ISKYLAB_TSAT=hygrometer python3 paper_figs_v3.py).
Overleaf: set the main document to main.tex, compiler pdfLaTeX.
