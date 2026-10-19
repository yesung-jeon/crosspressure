#!/usr/bin/env bash
# Build the manuscript and its appendix from the results (CPU only, a few minutes). Run after run_analysis.sh (Path B);
# run_analysis.sh calls it at the end. Every step asserts its numbers against sealed/ (or the sealed SEAL.json of each post/ folder).
#   1. post/members_by_era   descriptive follow-up for Section 7 and Tables B2, B6
#   2. post/benchmark        DW-NOMINATE and recognized-row checks for Section 4 and Table B7
#   3. paper/figures         Figures 1-3
#   4. paper/fill_intro.py   fills the abstract, introduction and Sections 2-7, Appendix A from the results
#   5. paper/make_appendix_b.py, paper/assemble.py   Appendix B and paper/manuscript.md
#   6. paper/build_pdf.py    paper/pdf/main.pdf and paper/pdf/appendix.pdf (needs Chrome or Edge; skipped if absent)
# The generated text files and figure PNGs are hashed before and after, and the script reports whether they are byte-identical
# to the shipped versions. PDFs embed a creation date and are not byte-compared.
# Usage: bash replication/build_paper.sh [--from-reproduced]   (--from-reproduced: fill_intro.py reads the reproduced outputs, not sealed/)
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"; cd "$R"
export PYTHONIOENCODING=utf-8
FILL_ARGS=("$@")
python paper/check_generated.py snapshot
python post/members_by_era/members_by_era.py
python post/benchmark/benchmark_recall.py
python paper/check_generated.py seals              # post/ outputs must match their SEAL.json
python paper/figures/fig1_map.py
python paper/figures/fig2_profiles.py
python paper/figures/fig3_packages.py
python paper/fill_intro.py "${FILL_ARGS[@]}"
python paper/make_appendix_b.py
python paper/assemble.py
python paper/build_pdf.py || echo "build_pdf.py failed (Chrome or Edge needed); the Markdown manuscript is complete"
python paper/check_generated.py compare
