#!/usr/bin/env bash
# Path A: full rerun from raw data and the public models (GPU). Usage:
#   bash replication/run_all.sh                 # all three models (many hours on a 16 GB GPU)
#   bash replication/run_all.sh qwen            # one model; the other models' shipped intermediates fill the joint steps
#   SMOKE=1 bash replication/run_all.sh qwen    # smoke test: tiny GPU runs into */smoke folders, no real outputs touched
# Needs: requirements-gpu.txt, Hugging Face access to meta-llama/Llama-3.1-8B-Instruct (the layer-selection judge is Llama for every
# model), data/fetch_data.py (Voteview) and the PAP file in data/agendas/ (README, "Data").
# Each step leaves logs/done_<name> and is skipped on a rerun; GPU steps retry with smaller batches (64 -> 32 -> 16; OLMo 32 -> 16 -> 8).
# Commands are the final calls behind the sealed results (kim records: TRACE.md); superseded steps are not included.
set -uo pipefail
R="$(cd "$(dirname "$0")" && pwd)"; export PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for d in congress_map data_generation; do [ -d "$R/../$d" ] || { echo "missing ../$d: replication/, congress_map/ and data_generation/ must sit side by side (README, Folder layout)"; exit 1; }; done
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(qwen llama olmo)
SM=""; [ "${SMOKE:-0}" = 1 ] && SM="--smoke"
LOG="$R/logs${SM:+_smoke}"; mkdir -p "$LOG"
step() {   # step <name> <dir> <batch-mode: none|std|olmo> <script> [args...]
  local name=$1 dir=$2 bm=$3; shift 3
  [ -f "$LOG/done_$name" ] && { echo "$(date +%H:%M) $name: done, skipped"; return 0; }
  local batches=(""); [ "$bm" = std ] && batches=(64 32 16); [ "$bm" = olmo ] && batches=(32 16 8)
  for b in "${batches[@]}"; do
    local extra=(); [ -n "$b" ] && extra=(--batch "$b")
    echo "$(date +%H:%M) $name ${extra[*]}"
    if (cd "$dir" && python "$@" "${extra[@]}" >> "$LOG/$name.log" 2>&1); then touch "$LOG/done_$name"; return 0; fi
    echo "$(date +%H:%M) $name failed (see $LOG/$name.log)"; sleep 60
  done
  echo "$(date +%H:%M) $name: FAILED, stopped"; exit 1; }
D="$R/STUDY_D_quadrant/code"; L="$R/STUDY_L_battery/code"
has() { local x; for x in "${MODELS[@]}"; do [ "$x" = "$1" ] && return 0; done; return 1; }
bm() { [ "$1" = olmo ] && echo olmo || echo std; }

[ -z "$SM" ] && step fresh "$R" none clear_intermediates.py "${MODELS[@]}"           # shipped outputs of these models aside: rerun every stage
[ -z "$SM" ] && step data "$R" none data/fetch_data.py
[ -z "$SM" ] && step issue_scales "$R/STUDY_B_polarization" none 00_data.py                 # economic and abortion roll-call scales
for m in "${MODELS[@]}"; do                                                                # model directions and the layer rule
  step "s1_$m" "$D" "$(bm $m)" s1_vectors.py --model "$m" $SM
  step "s2gen_$m" "$D" none s2_steer_gen.py --model "$m" $SM
  step "s2judge_$m" "$D" none s2_judge.py --models "$m" $SM
  step "s2rule_$m" "$D" none s2_rule.py --model "$m" $SM
  step "s3_$m" "$D" "$(bm $m)" s3_project.py --model "$m" --which main $SM               # legislator coordinates (Kim's step 3)
done
if [ -n "$SM" ]; then                                                                      # smoke: the two remaining GPU experiments
  for m in "${MODELS[@]}"; do step "s5_$m" "$D" none s5_inject.py --model "$m" --smoke; step "l1_$m" "$L" none l1_main.py --model "$m" --smoke; done
  echo "smoke test complete (GPU steps only)"; exit 0
fi
step s4 "$D" none s4_quadrant.py                                                          # acts 1-2, Qwen and Llama
step s4_olmo "$D" none s4_quadrant.py --models olmo --out s4_olmo
step s4b_select "$D" none s4b_select.py
step s4c_select "$D" none s4c_select.py
python "$R/check_selection.py" || exit 1                                                 # A12 codes are keyed to these selections
step s4b "$D" none s4b_positions.py
step s4b_olmo "$D" none s4b_positions.py --models olmo --out s4b_olmo
step s4c "$D" none s4c_positions.py                                                       # act 2 (A12): 15 legislators per quadrant
step s4c_olmo "$D" none s4c_positions.py --models olmo --out s4c_olmo
step s5_select "$D" none s5_select.py                                                     # act 3: stated profiles
for m in "${MODELS[@]}"; do step "s5_$m" "$D" none s5_inject.py --model "$m"; done
step s5_analyze "$D" none s5_analyze.py
step s5_analyze_olmo "$D" none s5_analyze.py --models olmo --out s5_olmo
for m in "${MODELS[@]}"; do                                                                # act 4: bundled choices (all 37 conditions)
  step "l0_$m" "$L" none l0_calibrate.py --model "$m"
  step "l1_$m" "$L" none l1_main.py --model "$m"
  step "l2_$m" "$L" none l2_analyze.py --model "$m"
done
for m in qwen llama olmo; do has "$m" || step "l2_$m" "$L" none l2_analyze.py --model "$m"; done   # other models: shipped l1 outputs (CPU)
step no_persona "$R" none post/no_persona/no_persona_effect.py
cd "$R" && python compare_to_sealed.py --path A --models "${MODELS[@]}"
# The manuscript is not rebuilt here: Path A matches verdicts, not digits, and build_paper.sh asserts digits. Build it with
# bash replication/run_analysis.sh (Path B), which ends with build_paper.sh.
