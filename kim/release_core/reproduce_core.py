#!/usr/bin/env python
"""Seal and reproduce the core results (steps 2-5 of kim/manuscript/RESEARCH_DESIGN.md v2). One linear script, CPU.

What it does, in order:
  1 copy the frozen intermediate outputs (generations, judge scores, projections) from uncertainty_map/out into a fresh run
    folder and record SHA-256 for them and for the fixed external inputs (Kim's results, Voteview, question banks)
  2 re-run the analysis scripts that produced the reported numbers, unchanged, in that folder (same estimand, one code path):
    04_analyze, 10_analyze_offset, 11_dynamics, 19_rep_and_time, 21_niche, 24_output_econ_magnitude, 26_analyze_mech_robust
  3 extract every number cited in steps 2-5 from the re-run outputs and compare it with the number in the original outputs
    (Kim K18: "matches the original by how much", not "it runs")
  4 --seal: write RELEASE.json (values, input and code hashes, environment) and CLAIMS.md (claims with values filled from outputs)
    default (no --seal): compare the fresh re-run with the sealed RELEASE.json and exit 1 on any mismatch
Tier 2 (regenerating the intermediates on a GPU: steps 02, 09, 18, 20, 23, 25) is not run here; sampled generations reproduce in
distribution, not token by token, so tier 2 is checked against these sealed values with tolerances, not exactly.
Usage:  python reproduce_core.py --seal      (first time)
        python reproduce_core.py             (verify later)
"""
import argparse, datetime, glob, hashlib, json, os, platform, shutil, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--seal", action="store_true"); ap.add_argument("--tol", type=float, default=1e-8); args = ap.parse_args()
HERE = Path(__file__).resolve().parent; KIM = HERE.parent; UM = KIM / "uncertainty_map"; SRC = UM / "out"; REPO = KIM.parent
RUN = HERE / ("rerun_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")); RUN.mkdir(parents=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(*a): print(datetime.datetime.now().strftime("%H:%M:%S"), *a, flush=True)

# ---------------- 1 inputs ----------------
INPUTS = ["member_generations.csv", "counterfactual_personas.csv", "counterfactual_generations.csv", "anonymous_cells.csv", "anonymous_generations.csv",
          "real_member_projections.csv", "offset_personas.csv", "offset_generations.csv", "t1_personas.csv", "t1_generations.csv", "time_direction.npy",
          "time_persons.csv", "time_panel.csv", "stance_validation.csv", "stance_counterfactual.csv", "stance_offset.csv", "stance_members.csv",
          "ordinary_by_question.csv"] + \
         [str(Path(p).relative_to(SRC)) for p in sorted(glob.glob(str(SRC / "corrected" / "*.csv")))] + \
         [str(Path(p).relative_to(SRC)) for p in sorted(glob.glob(str(SRC / "r3" / "*.csv"))) if Path(p).name.startswith(("judged_", "m2_", "r1_", "r3_"))]
manifest = {}
for rel in INPUTS:
    dst = RUN / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(SRC / rel, dst); manifest[rel] = sha(dst)
EXTERNAL = [REPO / "congress_map" / "results" / "temporal_member_scores.csv", REPO / "congress_map" / "results" / "vectors.pt", KIM / "data" / "HSall_members.csv",
            KIM / "1_data_question_bank.json", REPO / "data_generation" / "trait_data_extract" / "data_economic.json",
            REPO / "data_generation" / "trait_data_extract" / "data_social.json"] + sorted((KIM / "data" / "voteview").glob("[HS]*.csv"))
external = {str(p.relative_to(REPO)): sha(p) for p in EXTERNAL}
log(f"copied {len(manifest)} intermediate files; hashed {len(external)} external inputs")

# ---------------- 2 re-run the analyses ----------------
SCRIPTS = ["04_analyze.py", "10_analyze_offset.py", "11_dynamics.py", "19_rep_and_time.py", "21_niche.py", "24_output_econ_magnitude.py", "26_analyze_mech_robust.py"]
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore", MPLBACKEND="Agg")
for s in SCRIPTS:
    r = subprocess.run([sys.executable, str(UM / s), "--out", str(RUN)], cwd=UM, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (RUN / f"log_{s}.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0: log(f"FAILED {s}; see {RUN / ('log_' + s + '.txt')}"); sys.exit(2)
    log(f"re-ran {s}")
code = {s: sha(UM / s) for s in SCRIPTS + ["common.py"]} | {"reproduce_core.py": sha(__file__)}

# ---------------- 3 values cited in steps 2-5 ----------------
def J(base, name): return json.loads((base / name).read_text(encoding="utf-8"))
def get(d, *path):
    for p in path: d = d[p]
    return d
def peak_year(base):
    rows = [r for r in J(base, "rep_time_summary.json")["b_time"]["persons_by_year"] if r["half"] == "heldout"]
    return max(rows, key=lambda r: r["t_proj"])["year"]
V = {  # key: (step, claim, function(base) -> value)
 "s2_yearswap_internal_y1955": (2, "same legislator in 1955, internal social position vs own year", lambda b: get(J(b, "summary.json"), "C_counterfactual", "year_swap_y_vs_own_year", "y1955", "est")),
 "s2_yearswap_internal_y2015": (2, "same legislator in 2015, internal social position vs own year", lambda b: get(J(b, "summary.json"), "C_counterfactual", "year_swap_y_vs_own_year", "y2015", "est")),
 "s2_yearswap_output_y1955": (2, "same legislator in 1955, judge score on social answers vs own year", lambda b: get(J(b, "output_econ_magnitude.json"), "2_outputs", "yearswap_social", "y1955", "est")),
 "s2_yearswap_output_y2015": (2, "same legislator in 2015, judge score on social answers vs own year", lambda b: get(J(b, "output_econ_magnitude.json"), "2_outputs", "yearswap_social", "y2015", "est")),
 "s2_ordinary_internal_1955_minus_2023": (2, "ordinary persons 1955 vs 2023, internal social position (paired)", lambda b: get(J(b, "offset_summary.json"), "paired", "C7_nonpolitical_1955 - C4_nonpolitical", "y", "diff")),
 "s2_ordinary_output_1955_minus_2023": (2, "ordinary persons 1955 vs 2023, judge score on social answers", lambda b: get(J(b, "output_econ_magnitude.json"), "2_outputs", "ordinary_1955_minus_2023_social", "mean")),
 "s2_m2_ordinary_decade_ratio": (2, "decade wording / numeric year, ordinary persons", lambda b: get(J(b, "r3/mech_robust_summary.json"), "M2", "ordinary|decade", "ratio_to_numeric")),
 "s2_m2_ordinary_era_ratio": (2, "era wording / numeric year, ordinary persons", lambda b: get(J(b, "r3/mech_robust_summary.json"), "M2", "ordinary|era", "ratio_to_numeric")),
 "s2_m2_legislator_decade_ratio": (2, "decade wording / numeric year, legislators", lambda b: get(J(b, "r3/mech_robust_summary.json"), "M2", "legislator|decade", "ratio_to_numeric")),
 "s2_m2_legislator_era_ratio": (2, "era wording / numeric year, legislators", lambda b: get(J(b, "r3/mech_robust_summary.json"), "M2", "legislator|era", "ratio_to_numeric")),
 "s2_m2_employee_control_ratio": (2, "employee-number control / numeric year", lambda b: get(J(b, "r3/mech_robust_summary.json"), "M2", "ordinary|employee_control", "ratio_to_numeric")),
 "s2_tone_ordinary_sd": (2, "tone shift 1955 vs 2023, ordinary persons (SD units)", lambda b: get(J(b, "r3/mech_robust_summary.json"), "M4", "standardized_1955_vs_2023_ordinary", "tone")),
 "s2_ideology_ordinary_sd": (2, "social-ideology shift 1955 vs 2023, ordinary persons (SD units)", lambda b: get(J(b, "r3/mech_robust_summary.json"), "M4", "standardized_1955_vs_2023_ordinary", "social_ideology")),
 "s2_formality_ordinary_sd": (2, "formality shift 1955 vs 2023, ordinary persons (SD units)", lambda b: get(J(b, "r3/mech_robust_summary.json"), "M4", "standardized_1955_vs_2023_ordinary", "formality")),
 "s2_time_direction_heldout_r": (2, "time direction vs year, held-out persons (r)", lambda b: get(J(b, "rep_time_summary.json"), "b_time", "heldout_corr_tproj_year")),
 "s2_time_direction_peak_year": (2, "year at which the held-out time projection peaks", peak_year),
 "s2_wvs_yearswap_ratio": (2, "WVS-question year-swap shift / Kim-question shift", lambda b: get(J(b, "r3/mech_robust_summary.json"), "R1", "yearswap_1955_minus_2015", "ratio")),
 "s2_wvs_person_corr": (2, "person-level r, WVS vs Kim-question year-swap shift", lambda b: get(J(b, "r3/mech_robust_summary.json"), "R1", "yearswap_1955_minus_2015", "person_corr")),
 "s3_calendar_internal_kim_per_year": (3, "within-member calendar drift, internal social position per year (Kim questions)", lambda b: get(J(b, "dynamics_summary.json"), "R1_career", "y", "b_member_FE_plus_tenure", "tenure_years", "est")),
 "s3_calendar_internal_wvs_per_year": (3, "within-member calendar drift per year (WVS questions)", lambda b: get(J(b, "r3/mech_robust_summary.json"), "R1", "panel_within_member", "tenure", "est")),
 "s3_real_drift_internal_np1": (3, "within-member response to real roll-call change (Nokken-Poole dim1)", lambda b: get(J(b, "dynamics_summary.json"), "R1_career", "y", "b_member_FE_plus_tenure", "np1", "est")),
 "s3_calendar_output_per_decade": (3, "calendar drift in social answers per decade, net of the record", lambda b: get(J(b, "output_econ_magnitude.json"), "2_outputs", "members_social_calendar_per_decade_net_of_record", "est")),
 "s3_rep_corr_raw": (3, "Republican party-mean social position vs real ideology, uncorrected (r)", lambda b: get(J(b, "niche_summary.json"), "A_validation", "raw", "Republican_y_levels_r_np1")),
 "s4_yearswap_dw_units": (4, "1955 vs 2015 year swap in DW-NOMINATE units", lambda b: get(J(b, "output_econ_magnitude.json"), "5_magnitude", "yearswap_in_dim1_units")),
 "s4_party_gap_dw_2023": (4, "Republican - Democrat DW-NOMINATE gap, 2023", lambda b: get(J(b, "output_econ_magnitude.json"), "5_magnitude", "party_gap_dim1_2023")),
 "s4_real_rep_change_dw": (4, "real Republican change in DW-NOMINATE, 1947-2023", lambda b: get(J(b, "output_econ_magnitude.json"), "5_magnitude", "real_party_change_dim1_1947_2023", "Republican")),
 "s4_output_ratio": (4, "answers: year-swap shift / Republican - Democrat gap", lambda b: get(J(b, "output_econ_magnitude.json"), "5_magnitude", "outputs_ratio_yearswap_to_party_gap")),
 "s5_rep_corr_corrected": (5, "Republican party-mean social position vs real ideology, corrected (r)", lambda b: get(J(b, "niche_summary.json"), "A_validation", "corrected", "Republican_y_levels_r_np1")),
 "s5_baseline_corr_min": (5, "corrected Republican r, minimum over 8 baseline groups", lambda b: min(v["rep_corr_np1"] for k, v in J(b, "r3/mech_robust_summary.json")["R3"].items() if not k.startswith("_"))),
 "s5_baseline_corr_max": (5, "corrected Republican r, maximum over 8 baseline groups", lambda b: max(v["rep_corr_np1"] for k, v in J(b, "r3/mech_robust_summary.json")["R3"].items() if not k.startswith("_"))),
 "s5_rep_change_age70": (5, "Republican 1947-2023 change under the age-70 baseline", lambda b: get(J(b, "r3/mech_robust_summary.json"), "R3", "age70", "rep_change")),
 "s5_rep_change_south_rural": (5, "Republican 1947-2023 change under the Southern rural baseline", lambda b: get(J(b, "r3/mech_robust_summary.json"), "R3", "south_rural", "rep_change")),
 "s5_econ_output_ordinary_1955_minus_2023": (5, "ordinary persons 1955 vs 2023, judge score on economic answers (+ = market)", lambda b: get(J(b, "output_econ_magnitude.json"), "2_outputs", "ordinary_1955_minus_2023_economic", "mean")),
 "s5_econ_internal_slope_mean": (5, "mean per-century slope of internal economic position (+ = right) for ordinary persons", lambda b: sum(r["slope_per_century"] for r in J(b, "output_econ_magnitude.json")["3_economic"]["slopes_by_question"] if r["axis"] == "economic") / 5),
 "judge_auc_social": (0, "judge validation AUC, social", lambda b: get(J(b, "output_econ_magnitude.json"), "judge", "social", "auc")),
 "judge_auc_economic": (0, "judge validation AUC, economic", lambda b: get(J(b, "output_econ_magnitude.json"), "judge", "economic", "auc")),
}
rerun = {k: float(fn(RUN)) for k, (_, _, fn) in V.items()}
original = {k: float(fn(SRC)) for k, (_, _, fn) in V.items()}
ok = lambda a, b: abs(a - b) <= args.tol * max(1.0, abs(b))
cmp_orig = {k: dict(rerun=rerun[k], original=original[k], match=ok(rerun[k], original[k])) for k in V}
n_bad = sum(not c["match"] for c in cmp_orig.values()); log(f"re-run vs original outputs: {len(V) - n_bad}/{len(V)} match")

# ---------------- 4 seal or verify ----------------
rel = HERE / "RELEASE.json"
if args.seal:
    if n_bad: log("not sealed: the re-run does not reproduce the original outputs"); sys.exit(1)
    import numpy, pandas, scipy, statsmodels
    R = dict(sealed=datetime.datetime.now().isoformat(timespec="seconds"), scope="kim/manuscript/RESEARCH_DESIGN.md v2, steps 2-5 (step 1 excluded)",
             model="Qwen2.5-7B-Instruct (int8) generator; Qwen2.5-3B-Instruct judge", status="exploratory; one model",
             values={k: dict(value=rerun[k], step=V[k][0], claim=V[k][1]) for k in V}, intermediate_inputs_sha256=manifest, external_inputs_sha256=external,
             code_sha256=code, environment=dict(python=platform.python_version(), numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__,
             statsmodels=statsmodels.__version__, platform=platform.platform()), rerun_folder=RUN.name)
    rel.write_text(json.dumps(R, indent=1), encoding="utf-8")
    L = ["# Core claims with sealed values (generated by reproduce_core.py from RELEASE.json; do not edit by hand)", ""]
    for step in (2, 3, 4, 5, 0):
        L.append(f"## {'Step ' + str(step) if step else 'Judge validation'}")
        L += [f"- {V[k][1]}: **{rerun[k]:.3f}**  (`{k}`)" if not k.endswith("peak_year") else f"- {V[k][1]}: **{int(rerun[k])}**  (`{k}`)" for k in V if V[k][0] == step]
        L.append("")
    (HERE / "CLAIMS.md").write_text("\n".join(L), encoding="utf-8"); log(f"sealed {len(V)} values -> {rel}")
else:
    if not rel.exists(): log("no RELEASE.json; run with --seal first"); sys.exit(1)
    R = json.loads(rel.read_text(encoding="utf-8")); bad = []
    for k, v in R["values"].items():
        if not ok(rerun[k], v["value"]): bad.append((k, rerun[k], v["value"]))
    changed = [k for k, h in R["intermediate_inputs_sha256"].items() if manifest.get(k) != h]
    log(f"verify vs sealed: {len(R['values']) - len(bad)}/{len(R['values'])} match; changed inputs: {changed or 'none'}")
    for k, a, b in bad: log(f"  MISMATCH {k}: rerun {a} vs sealed {b}")
    (RUN / "VERIFY.json").write_text(json.dumps(dict(mismatches=bad, changed_inputs=changed), indent=1), encoding="utf-8")
    sys.exit(1 if bad else 0)
(RUN / "COMPARE_WITH_ORIGINAL.json").write_text(json.dumps(cmp_orig, indent=1), encoding="utf-8")
