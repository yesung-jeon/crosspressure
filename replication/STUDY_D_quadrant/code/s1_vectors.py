#!/usr/bin/env python
"""S1 (PLAN.md): run Kim's congress_map/01_extract_vectors.py unchanged (FP8 via a patched from_pretrained) and keep its
per-layer raw differences (`diffs`) as diffs_all.pt. B1 for Qwen: cosine with Kim's results/vectors.pt at layer 17.
Usage: python s1_vectors.py --model qwen [--smoke]"""
import argparse, json, runpy, sys
import torch
import dcommon as dc

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(dc.MODELS)); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--batch", default="64", help="Kim generation batch (lowered by the chain on retry)")
a = ap.parse_args()
out = dc.model_dir(a.model) / ("s1_smoke" if a.smoke else "s1"); out.mkdir(parents=True, exist_ok=True)
if (out / "diffs_all.pt").exists() and not a.smoke:
    dc.log("S1 done, skipped"); sys.exit(0)
dc.ram_guard()
dc.patch_from_pretrained()
argv = ["01_extract_vectors.py", "--repo_root", str(dc.REPO), "--model", dc.MODELS[a.model], "--layer", "17", "--out", str(out)]
argv += ["--rollouts", "1", "--max_new", "24"] if a.smoke else ["--rollouts", "5", "--max_new", "256", "--batch", a.batch]
sys.argv = argv
g = runpy.run_path(str(dc.CMAP / "01_extract_vectors.py"), run_name="__main__")
diffs = {k: v.float().cpu() for k, v in g["diffs"].items()}
assert diffs["economic"].shape == diffs["social"].shape and diffs["economic"].shape[0] == g["NL"], diffs["economic"].shape
torch.save({"diffs": diffs, "model": dc.MODELS[a.model], "n_layers_plus_emb": int(g["NL"])}, out / "diffs_all.pt")
dc.log("saved", out / "diffs_all.pt", tuple(diffs["economic"].shape))
if a.model == "qwen":
    K = torch.load(dc.CMAP / "results" / "vectors.pt"); L = int(K["layer"])
    ue = diffs["economic"][L] / diffs["economic"][L].norm(); us = diffs["social"][L] / diffs["social"][L].norm()
    b1 = {"layer": L, "cos_raw_e": float(ue @ K["raw_e"].float()), "cos_raw_s": float(us @ K["raw_s"].float())}
    b1["pass"] = bool(b1["cos_raw_e"] >= 0.90 and b1["cos_raw_s"] >= 0.90)
    dc.atomic_json(b1, out / "B1.json"); dc.log("B1", b1)
