#!/usr/bin/env bash
# Path B: reproduce every act 1-4 number from the shipped intermediates, then build the manuscript (CPU only, about 10-30 minutes).
# Run from anywhere: bash replication/run_analysis.sh
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"; cd "$R"
for d in congress_map data_generation; do [ -d "$R/../$d" ] || { echo "missing ../$d: replication/, congress_map/ and data_generation/ must sit side by side (README, Folder layout)"; exit 1; }; done
export PYTHONIOENCODING=utf-8
python data/fetch_data.py --members-only
cd "$R/STUDY_D_quadrant/code"
python s4_quadrant.py                                  # act 1 (A3, A4, A6) and act 2 (A9): Qwen, Llama
python s4_quadrant.py --models olmo --out s4_olmo      #   OLMo
python s4b_positions.py                                # S4b (30 deepest legislators; feeds S4c's text codes)
python s4b_positions.py --models olmo --out s4b_olmo
python s4c_positions.py                                # act 2 (A12): 60 legislators chosen by votes, 15 per quadrant
python s4c_positions.py --models olmo --out s4c_olmo
python s5_analyze.py                                   # act 3 (A'1, A'3, A'4): stated profiles
python s5_analyze.py --models olmo --out s5_olmo
cd "$R/STUDY_L_battery/code"
for m in qwen llama olmo; do python l2_analyze.py --model "$m"; done   # act 4 (B1, B2, B5): bundled choices
cd "$R"
python post/no_persona/no_persona_effect.py            # act 4: no-persona effect with two-way item bootstrap
python compare_to_sealed.py --path B
bash "$R/build_paper.sh"                               # manuscript, appendix, figures and PDFs from the checked results
