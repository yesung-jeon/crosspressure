# From the introduction to the code and data

Each result sentence in `introduction.md` is listed with the number it states, the script that produces it, and the file it is read from.
Numbers are filled by `fill_intro.py` (values and their source in `intro_values.json`); `bash run_analysis.sh` reproduces every file below
from the shipped intermediates, and `compare_to_sealed.py` checks them against `sealed/`.

| Introduction sentence | Value key | Produced by | Read from |
|---|---|---|---|
| Economic coordinate correlates with the economic vote scale; social coordinate with the abortion vote scale | `r_econ`, `r_abort` | `STUDY_D_quadrant/code/s4_quadrant.py` (Qwen, Llama; `--models olmo --out s4_olmo` for OLMo) | `STUDY_D_quadrant/out/s4*/anchors.json`: `r_x_economic_scale_within_cc`, `r_y_abortion_scale_within_cc` |
| Cross-pressured members account for a share of member-Congress rows | `cross_share` | same | `out/s4*/vote_vs_map.json`: rows of `confusion_vote_rows_map_cols` for EL-ST and ER-SP over `n` |
| Map places aligned rows / cross-pressured rows in their voting quadrant | `recall_aligned`, `recall_cross`, `n_rows` | same | `out/s4*/vote_vs_map.json`: `recall_by_vote_quadrant` |
| Legislators chosen only by their votes, 15 per quadrant, placed in their own quadrant | `a12_n`, `a12_aligned`, `a12_cross` | `s4c_select.py` (selection from votes only) then `s4c_positions.py` | `out/s4c*/S4C.json`: `placed` |
| Stated profiles: consistency and quadrant arrival lower for cross-pressured profiles | `h3`, `h4` | `s5_select.py`, `s5_inject.py` (GPU), `s5_analyze.py` | `out/s5*/S5_RESULTS.json`: `hyp.H3_C2`, `hyp.H4_C3hit` |
| Choice entropy, crossed minus party-line packages, by party | `b_qwen`, `b_olmo`, `b_llama` | `STUDY_L_battery/code/l1_main.py` (GPU), `l2_analyze.py` | `STUDY_L_battery/out/<model>/RESULTS.json`: `blockB_natural_cross_pressure.tests.B1_Democrat`, `B2_Republican` |
| Same contrast without any persona, with intervals that resample policy passages | `np_qwen`, `np_olmo`, `np_llama` | `post/no_persona/no_persona_effect.py` | `post/no_persona/no_persona_package_effect.json`: `interval_2026_10_18` (`est`, `ci95_pigeonhole`) |

Terms used in the introduction and in the code:

| Introduction | Code |
|---|---|
| economic and social direction | `orth_e`, `orth_s` in `vectors.pt` (Kim's `congress_map/01_extract_vectors.py`, run unchanged; layer from `s2_rule.py`) |
| place the member on the two directions | `econ_orth`, `social_orth` in `s3/temporal_member_scores.csv` (`congress_map/03_project_legislators.py`; no party in the prompt) |
| voting quadrant (aligned: EL-SP, ER-ST; cross-pressured: EL-ST, ER-SP) | median splits of the economic and abortion roll-call scales within Congress and chamber, 93rd Congress on (`s4_quadrant.py`; scales from `STUDY_B_polarization/00_data.py`) |
| stated profile | `s5_inject.py` modes I1–I5 (four profiles: EL-SP, ER-ST, EL-ST, ER-SP) |
| crossed vs party-line packages | `family` = `reversed` vs `party_like` in Study L units (6 economic x 9 social passages = 54 pairs) |
| choice entropy (bits) | binary entropy of the first-answer probabilities, `f_common.hbits` |

## Section 3 (design), draft v1 2026-10-19

Filled values (fill_intro.py, sealed files): layer_qwen/llama/olmo = `s4/anchors.json`, `s4_olmo/anchors.json` "layer"; n_map_rows = anchors "n_rows" (asserted equal across models); recall_range = anchors "recall"; n_rows, cross_share = `vote_vs_map.json`; a12_n = `s4c/S4C.json` "n"; s5_n = `S5_RESULTS.json` "n_conditions" (asserted equal across models); np_n_pairs = `no_persona_package_effect.json` "n_pairs".

Design constants (read from code, not results):
| Text | Value | Source |
|---|---|---|
| extraction questions, instruction pairs, answers per question | 20, 5, 5 | congress_map/01_extract_vectors.py (`qs[:20]`, `instructions`), STUDY_D_quadrant/code/s1_vectors.py (`--rollouts 5`, `--max_new 256`) |
| answer filter | at least 15 tokens, no refusal opening | 01_extract_vectors.py `ok` |
| steering strengths, held-out questions | 10 (0.05–0.60), questions 20–29 | s2_steer_gen.py docstring |
| judge and scale | Llama-3.1-8B-Instruct FP8, 0–100 | s2_judge.py |
| layer rule | best mean rating across strengths, minimum rank sum | s2_rule.py |
| member prompt, held-out questions | quoted text; 5 + 5 (questions 20–24); last prompt token; minus no-persona | congress_map/03_project_legislators.py |
| party recall | one-word answer containing the party name | 03_project_legislators.py `known` |
| economic roll calls | PAP major topics 1, 5, 6, 7, 13 | STUDY_B_polarization/issues.py `RC_ONLY` |
| abortion roll calls | regex ABORTION|UNBORN | issues.py `ISSUES["abortion"]` |
| scale rules | minority ≥ 10%, ≥ 3 roll calls, ≥ 3 votes, first PC, orientation rule | STUDY_B_polarization/00_data.py docstring |
| quadrants | median splits within Congress × chamber, 93rd Congress on | s4_quadrant.py lines 78–87 |
| legislator check | 10 deepest + 5 random per vote quadrant; mean median-centred coordinates | s4c_select.py, s4c_positions.py |
| profiles and modes | quoted positions; I1–I5 | s5_inject.py `EP`, `SP`, conditions loop |
| single-issue questions | 16 (8 + 8); shown examples excluded | cp3_items.json; s5_inject.py `exclude` |
| generation length (generation uncertainty) | up to 64 new tokens, greedy | s5_inject.py `GenerationConfig(max_new_tokens=64)` |
| S5 minima, bootstrap, rule | +0.05, +0.05, −0.05, −0.10; 2,000 within mode × profile; CI excl. 0, Holm < .05, minimum | s5_analyze.py `MIN`, `NB`, `boot`, `supported` |
| personas, passages | 20 (10 D, 10 R) + no-persona arm; 6 economic × 9 social; two orders | l1_main.py docstring; j_common.py `items` |
| package entropy, contrast, inference | normalized A/B binary entropy (bits), mean over orders; persona × topic fixed effects; CR1 by persona; wild cluster bootstrap 9,999 Rademacher; Holm (B1, B2); minimum +0.05 | l2_analyze.py `units`, `contrast`, B block |
| decided threshold | abs(P(A) − 0.5) ≥ 0.10 | l2_analyze.py `decided` |
| no-persona interval | two-way (pigeonhole) bootstrap, 10,000 draws | post/no_persona/no_persona_effect.py docstring |
| unreviewed passages | 6 of 9 social topics have origin "Study J" | j_common.py items `origin` |

## Section 4 (map), draft v1 2026-10-19

| Key / element | Source |
|---|---|
| r_dim1 | anchors "r_x_dim1_by_congress_mean" (three models) |
| r_econ, r_abort | anchors "r_x_economic_scale_within_cc", "r_y_abortion_scale_within_cc" (also in the introduction) |
| gap_x, gap_y | anchors "party_gap_x_d", "party_gap_y_d" (mean difference over the recognized-sample SD); assert economic > social in every model |
| r_cross_track, qwen_r_y_econ, qwen_r_y_abort | anchors "r_y_economic_scale_within_cc", "r_x_abortion_scale_within_cc", "r_y_abortion_scale_within_cc"; assert Qwen social follows economic more |
| recall_aligned, recall_cross, agree | vote_vs_map "recall_by_vote_quadrant", "agreement" |
| cross_to_aligned, mapped_aligned | vote_vs_map "confusion_vote_rows_map_cols" |
| elst_minus_ersp | vote_vs_map recall EL-ST minus ER-SP; assert positive in every model |
| a12_n, a12_aligned, a12_cross | S4C.json "placed", "n" |
| Figure 1 | paper/figures/fig1_map.py: (a) shipped Qwen map (STUDY_D_quadrant/out/qwen/s3) and selection (out/s4c/selection.csv), placements recomputed as in s4c_positions.py and asserted equal to sealed S4C.json; (b) vote_vs_map.json; (c) S4C.json. Labels: two members with the largest depth in each cross-pressured quadrant (fixed rule) |

## Section 5 (stated profiles), draft v1 2026-10-19

| Key / element | Source |
|---|---|
| c2_aligned, c2_cross, h3 | S5_RESULTS.json hyp H3_C2 "aligned_mean", "cross_mean", "est" |
| hit_aligned, hit_cross, h4 | hyp H4_C3hit "aligned_mean", "cross_mean", "est" |
| h1_llama, h1_olmo, h1_qwen | hyp H1_C1a "est", "ci95"; assert supported in Llama and OLMo only |
| h2, h2_min | hyp H2_C4 "est", "min_effect"; assert interval above zero and estimate below the minimum in every model |
| n_mode_cells, hit_neg_cells | "per_mode_cross_minus_aligned"; assert consistency gap negative in every model x way |
| base-prompt sensitivity sentence | "sensitivity_base_cluster_A14"; assert same zero-exclusion as the primary interval for every test |
| Figure 2 | paper/figures/fig2_profiles.py from the same files |

## Section 6 (package choices), draft v1 2026-10-19

| Key / element | Source |
|---|---|
| own_qo, own_llama_d, own_llama_r | STUDY_L_battery/out/{m}/RESULTS.json blockB tests "own_package_at_NONE_by_party"; assert Llama below every Qwen/OLMo value |
| H_party_qo, H_cross_qo, H_party_llama, H_cross_llama | tests "entropy_at_NONE_by_party_family" (stored as a Python repr string, parsed with ast.literal_eval); assert direction per model |
| b_qwen, b_olmo, b_llama | tests B1_Democrat, B2_Republican "est" (also in the introduction); assert Llama intervals below zero and not supported |
| np_qwen, np_olmo, np_llama | no_persona_package_effect.json "interval_2026_10_18" est, ci95_pigeonhole; assert Llama interval contains zero, Qwen/OLMo above zero |
| esh_*, dec_*, esh_cells_over_half | tests "economic_dominance_share_by_party", "decided_units_by_party"; assert five of six above one half |
| Figure 3 | paper/figures/fig3_packages.py from the same files (interval for B1/B2 = percentile wild cluster bootstrap, l2_analyze.py wcb) |

## Section 7 (discussion), draft v1 2026-10-19

Values from the descriptive follow-up `post/members_by_era/members_by_era.json` and `members_table.csv` (added 2026-10-19 at the author's request, after the sealed results; descriptive only; outputs hashed in `post/members_by_era/SEAL.json`; placements asserted equal to sealed S4C.json and row recall equal to sealed vote_vs_map.json inside the script). recall_cross from the sealed vote_vs_map.json.
| Key / statement | Source and assert |
|---|---|
| elst_dem, elst_all, ersp_rep, ersp_all | rows.qwen.cross_rows_party_by_quadrant (vote rows are the same for every model) |
| cross_none, cross_all3, aligned_all3 | members.cross_placed_by_k_models, aligned_placed_by_k_models |
| named senators; "only Randolph by all three" | members.senators_cross; assert the set of nine names and that only Randolph is placed by all three |
| Packwood and Huddleston in ER-ST in all three maps; Eagleton placed by Qwen only, EL-SP in Llama and OLMo | members_table.csv; asserts in fill_intro.py |
| rep_to_erst, dem_to_elsp_lo, dem_to_elsp_oo | members.misplaced_destination (OLMo excludes members it does not recognize) |
| cross_1970s, share_70s, share_10s, ra_70s_10s, rc_range | members.cross_first_decade; rows.<model>.by_decade (118th Congress grouped with 2010-2019, fixed before writing); assert aligned recall rises in every model |
Rule for naming: the senators among the 30 vote-selected cross-pressured legislators (fixed before writing), plus members already labeled in Figure 1.
