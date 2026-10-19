#!/usr/bin/env python
"""S3 (PLAN.md): run Kim's congress_map/03_project_legislators.py unchanged (FP8 via patched from_pretrained).
--which main : vectors.pt (rule layer), all members 80-118 -> out/<model>/s3/temporal_member_scores.csv
--which L17  : (Qwen, B3) vectors_L17.pt -> out/qwen/s3_L17/
--which B2   : (Qwen) Kim's own results/vectors.pt on 2,000 seeded member-Congress rows, no recall -> out/qwen/B2/ + B2.json
Usage: python s3_project.py --model qwen --which main [--smoke]"""
import argparse, runpy, sys
import numpy as np, pandas as pd
import dcommon as dc

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--which", default="main", choices=["main", "L17", "L16", "B2"])
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--batch", default="64", help="Kim projection batch (lowered by the chain on retry)")
a = ap.parse_args()
md = dc.model_dir(a.model)
sub = {"main": "s3", "L17": "s3_L17", "L16": "s3_L16", "B2": "B2"}[a.which] + ("_smoke" if a.smoke else "")
out = md / sub; out.mkdir(parents=True, exist_ok=True)
if (out / "temporal_member_scores.csv").exists():
    dc.log(f"S3 {sub} done, skipped"); sys.exit(0)
vec = {"main": md / ("vec_smoke" if a.smoke else "") / "vectors.pt", "L17": md / "vectors_L17.pt", "L16": md / "vectors_L16.pt", "B2": dc.CMAP / "results" / "vectors.pt"}[a.which]
assert vec.exists(), vec
voteview = dc.MEMBERS; extra = []
if a.which == "B2" or a.smoke:
    m = pd.read_csv(dc.MEMBERS, low_memory=False)
    m = m[m.congress.between(80, 118) & m.chamber.isin(["House", "Senate"]) & m.nominate_dim1.notna()].drop_duplicates(["congress", "icpsr"])
    n = 40 if a.smoke else 2000
    m = m.sample(n=n, random_state=dc.seed_for("D-B2") % 2**32)
    voteview = out / "members_subset.csv"; m.to_csv(voteview, index=False)
    if a.which == "B2": extra = ["--no_recall"]
dc.ram_guard()
dc.patch_from_pretrained()
sys.argv = ["03_project_legislators.py", "--repo_root", str(dc.REPO), "--model", dc.MODELS[a.model], "--vectors", str(vec),
            "--voteview", str(voteview), "--out", str(out), "--batch", a.batch] + extra
runpy.run_path(str(dc.CMAP / "03_project_legislators.py"), run_name="__main__")
if a.which == "B2":
    new = pd.read_csv(out / "temporal_member_scores.csv"); kim = pd.read_csv(dc.CMAP / "results" / "temporal_member_scores.csv")
    j = new.merge(kim, on=["congress", "icpsr"], suffixes=("_fp8", "_kim"), validate="one_to_one")
    assert len(j) == len(new), (len(j), len(new))
    b2 = {"n": int(len(j)), "r_econ_orth": float(np.corrcoef(j.econ_orth_fp8, j.econ_orth_kim)[0, 1]),
          "r_social_orth": float(np.corrcoef(j.social_orth_fp8, j.social_orth_kim)[0, 1]),
          "mean_abs_diff_econ": float((j.econ_orth_fp8 - j.econ_orth_kim).abs().mean()), "sd_kim_econ": float(j.econ_orth_kim.std()),
          "mean_abs_diff_social": float((j.social_orth_fp8 - j.social_orth_kim).abs().mean()), "sd_kim_social": float(j.social_orth_kim.std())}
    b2["pass"] = bool(b2["r_econ_orth"] >= 0.95 and b2["r_social_orth"] >= 0.95)
    dc.atomic_json(b2, out / "B2.json"); dc.log("B2", b2)
