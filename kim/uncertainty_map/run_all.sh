#!/bin/bash
# Full pipeline: Kim's congress_map results -> persona generations -> counterfactual and anonymous personas -> analyses A-E
# -> uncertainty map (HTML) + GIF. Run from this folder on one GPU (>= 12 GB free with --quant int8; tested on RTX 5060 Ti 16 GB).
# GPU steps (int8, RTX 5060 Ti): step 1 ~30 min (952 new member-Congresses), step 2 ~10 min, step 3 ~25 min, step 7 ~12 min. Steps 4-6 are CPU, a few minutes.
# For bf16 on a larger GPU pass QUANT=bf16 (e.g. QUANT=bf16 bash run_all.sh) and use a fresh OUT folder.
set -euo pipefail
# each GPU step writes $OUT/stepN.done when finished; an interrupted step resumes by batch on the next run
cd "$(dirname "$0")"
QUANT=${QUANT:-int8}; OUT=${OUT:-out}
mkdir -p "$OUT" logs
[ -f "$OUT/step1.done" ] || python 01_generate_members.py --out "$OUT" --quant "$QUANT"
[ -f "$OUT/step2.done" ] || python 02_counterfactual.py --out "$OUT" --quant "$QUANT"
[ -f "$OUT/step3.done" ] || python 03_anonymous_personas.py --out "$OUT" --quant "$QUANT"
[ -f "$OUT/step7.done" ] || python 07_left_of_default.py --out "$OUT" --quant "$QUANT"
[ -f "$OUT/step9.done" ] || python 09_persona_offset.py --out "$OUT" --quant "$QUANT"
python 12_get_rollcalls.py
[ -f "$OUT/step13.done" ] || python 13_vote_prediction.py --out "$OUT" --quant "$QUANT"
[ -f "$OUT/step14.done" ] || python 14_default_swap.py --out "$OUT" --quant "$QUANT"
[ -f "$OUT/step18.done" ] || python 18_time_direction.py --out "$OUT" --quant "$QUANT"
[ -f "$OUT/step20.done" ] || python 20_corrected_map.py --out "$OUT" --quant "$QUANT"
[ -f "$OUT/step23.done" ] || python 23_output_stance.py --out "$OUT"
[ -f "$OUT/step25.done" ] || python 25_mechanism_robustness.py --out "$OUT"
python 04_analyze.py --out "$OUT"
python 05_build_map.py --out "$OUT"
python 06_make_gif.py --out "$OUT"
python 08_analyze_t1.py --out "$OUT"
python 10_analyze_offset.py --out "$OUT"
python 11_dynamics.py --out "$OUT"
python 15_topics.py --out "$OUT"
python 16_analyze_votes_swap.py --out "$OUT"
python 17_representation.py --out "$OUT"
python 19_rep_and_time.py --out "$OUT"
python 21_niche.py --out "$OUT"
python 22_build_corrected_map.py --out "$OUT"
python 24_output_econ_magnitude.py --out "$OUT"
python 26_analyze_mech_robust.py --out "$OUT"
echo "done: $OUT/uncertainty_map.html  $OUT/uncertainty_map.gif  $OUT/summary.json"
