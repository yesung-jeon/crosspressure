# Study A core: from Junsol Kim's congress_map to the sealed core claims

Project overview (problem, claim, methods, findings): `OVERVIEW.md`.

One command after Kim's pipeline: `bash kim/study_a_core/run_core.sh` (int8; `QUANT=bf16 OUT=out_core_bf16` on a larger GPU).

| stage | what | why it is needed (claim steps of manuscript/RESEARCH_DESIGN.md v2) |
|---|---|---|
| 0 | preflight: Kim's `results/vectors.pt`, `results/temporal_member_scores.csv`, Voteview member file | inputs |
| 1 | GPU steps 01, 02, 03, 07, 09, 18, 20, 23, 25 (resumable, `stepN.done`) | 02 year swap, 09 ordinary persons, 18 time direction (step 2); 01 + 23 answers and judge (steps 2-4); 20 same-year correction (steps 3, 5); 25 wording, tone, WVS, baseline groups (steps 2, 5). 03 and 07 are kept only because 04_analyze and 10_analyze_offset read their files |
| 2 | CPU analyses 04, 10, 11, 19, 21, 24, 26 | the seven scripts behind the 36 sealed values |
| 3 | `compare_to_seal.py` | each value vs `release_core/RELEASE.json`: within max(0.05, 30%) with the sealed sign (sign not required for the three "about zero" claims); judge AUC >= 0.80 |

Left out (exploratory, not in the core claims): steps 05, 06, 08, 13, 14, 15, 16, 17, 22.
Checks done 2026-10-05: shell syntax; `compare_to_seal.py` on the sealed run gives 36/36 (`sanity_check_on_sealed_run.csv`);
`release_core/reproduce_core.py` still 36/36 after the step-1 change. Not done: a fresh end-to-end run (4-5 h on this PC).
Change made for fresh checkouts: `uncertainty_map/01_generate_members.py` starts empty when the local exploration cache is absent.
