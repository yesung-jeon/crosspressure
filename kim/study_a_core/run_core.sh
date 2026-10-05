#!/bin/bash
# Study A core: from Junsol Kim's congress_map outputs to the sealed core claims (steps 2-5), only the steps those claims need.
#
# Prerequisite (Kim's pipeline, run once):  cd congress_map && bash run_all.sh   then copy out/vectors.pt and
#   out/temporal_member_scores.csv to congress_map/results/ (as Kim's README says); data/HSall_members.csv from his step 2.
# Run:   bash kim/study_a_core/run_core.sh              (int8, fits a 16 GB GPU)
#        QUANT=bf16 OUT=out_core_bf16 bash kim/study_a_core/run_core.sh     (larger GPU)
# Each GPU step writes $OUT/stepN.done and is skipped next time; an interrupted step resumes by batch.
# Rough time on an RTX 5060 Ti (int8): 4-5 hours in total; most of it is steps 1, 3 and 20.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; KIM="$(dirname "$HERE")"; REPO="$(dirname "$KIM")"; UM="$KIM/uncertainty_map"
QUANT=${QUANT:-int8}; OUT=${OUT:-out_core}          # a fresh folder; the sealed run lives in uncertainty_map/out
export PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 0 preflight: Kim's outputs and the Voteview member file ----
for f in vectors.pt temporal_member_scores.csv; do
  if [ ! -f "$REPO/congress_map/results/$f" ]; then
    if [ -f "$REPO/congress_map/out/$f" ]; then echo "copy congress_map/out/$f to congress_map/results/ first (Kim's README)"; else echo "missing congress_map/results/$f: run congress_map/run_all.sh first"; fi; exit 1; fi
done
if [ ! -f "$KIM/data/HSall_members.csv" ]; then mkdir -p "$KIM/data"; cp "$REPO/congress_map/data/HSall_members.csv" "$KIM/data/" || { echo "missing HSall_members.csv (congress_map step 2)"; exit 1; }; fi
cd "$UM"; mkdir -p "$OUT" logs
step() { local n=$1; shift; [ -f "$OUT/step$n.done" ] && { echo "step $n: done, skipped"; return; }; echo "step $n: $(date +%H:%M)"; python "$@" --out "$OUT" > "logs/core_step$n.log" 2>&1; }

# ---- 1 GPU: generations and projections the core analyses read ----
step 1  01_generate_members.py --quant "$QUANT"      # legislator answers (judged in step 23: calendar drift in answers, step 3 of the claim)
step 2  02_counterfactual.py --quant "$QUANT"        # same legislator, prompt year swapped (claim step 2; magnitude, step 4)
step 3  03_anonymous_personas.py --quant "$QUANT"    # read by 04_analyze (not cited); also writes real_member_projections.csv
step 7  07_left_of_default.py --quant "$QUANT"       # read by 10_analyze_offset (not cited)
step 9  09_persona_offset.py --quant "$QUANT"        # ordinary persons in 2023 and 1955 (claim step 2)
python 12_get_rollcalls.py > logs/core_step12.log 2>&1   # Voteview roll calls for 4 Congresses (used by 19)
step 18 18_time_direction.py --quant "$QUANT"        # time direction from ordinary persons, held-out check (claim step 2)
step 20 20_corrected_map.py --quant "$QUANT"         # every member re-projected with the same-year ordinary-person origin (claim steps 3, 5)
step 23 23_output_stance.py                          # blind judge on answers (claim steps 2-4)
step 25 25_mechanism_robustness.py                   # wording (M2), tone (M4), WVS questions (R1), 8 baseline groups (R3)

# ---- 2 CPU: the seven analyses behind the 36 sealed values ----
for s in 04_analyze 10_analyze_offset 11_dynamics 19_rep_and_time 21_niche 24_output_econ_magnitude 26_analyze_mech_robust; do
  echo "analysis $s"; python $s.py --out "$OUT" > "logs/core_$s.log" 2>&1
done

# ---- 3 compare with the sealed values (tolerances: sampled generations reproduce in distribution only) ----
python "$HERE/compare_to_seal.py" --out "$UM/$OUT"
