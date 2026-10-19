#!/usr/bin/env python
"""l0 (PLAN addendum A1): per-model behavioural norm calibration. Pure pushes E+, E-, S+, S- at fractions f of mu on the 15 single-issue
items (2 orders) for the first Democrat and the first Republican persona; on-axis shift = mean over the two pushes of an axis and the two
personas of |P(own side) - P(own side at NONE)| on that axis's single issues. f_H = the smallest grid fraction whose on-axis shift,
averaged over the two axes, is >= TARGET (0.25; Qwen at 0.28 mu reaches it); f_L = f_H / 1.4. Qwen keeps 0.28 / 0.20 (Study K) and
the grid is written for comparison only. -> out/<model>/main/norms.json. Usage: python l0_calibrate.py --model llama"""
import argparse
import numpy as np, pandas as pd, torch
import l_common as lc

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(lc.MODELS)); a = ap.parse_args()
od = lc.out_dir(a.model, "main"); GRID = [0.14, 0.20, 0.28, 0.40, 0.56, 0.80, 1.12]; TARGET = 0.25
if (od / "norms.json").exists(): lc.log("norms.json exists, skipped"); raise SystemExit
L = lc.layer(a.model); tok, model = lc.load(a.model); oe, os_ = lc.directions(a.model, L)
P = lc.personas("confirm"); people = [next(p for p in P if p["party"] == "Democrat"), next(p for p in P if p["party"] == "Republican")]
_, singles = lc.items()
rows, texts = [], []
for p in people:
    for s in singles:
        for order in (0, 1):
            A, B = (s["pkgL"], s["pkgR"]) if order == 0 else (s["pkgR"], s["pkgL"])
            rows.append(dict(pid=p["pid"], own=p["own"], axis=s["axis"], order=order)); texts.append(lc.chat(tok, lc.fc.choice_msgs(p["system"], A, B)))
R = pd.DataFrame(rows); mu = lc.gc.mu_at(model, tok, texts[:64], L); PU = lc.Pusher(mu, oe, os_)
def run(W):
    with lc.Steer(model, L) as steer: PA, PB = lc.gc.score_final(tok, model, steer, texts, W)
    pf = np.where(R.order == 0, PA, PB) / (PA + PB); po = np.where(R.own > 0, pf, 1 - pf)          # P(own side): left for Democrats, right for Republicans
    return R.assign(po=po)
base = run(torch.zeros(len(texts), oe.shape[0])); b = base.groupby(["pid", "axis"]).po.mean()
grid = []
for f in GRID:
    sh = {}
    for d, ax in (("Ep", "econ"), ("Em", "econ"), ("Sp", "soc"), ("Sm", "soc")):
        r = run((f * mu * PU.dirs[d]).repeat(len(texts), 1)); m = r.groupby(["pid", "axis"]).po.mean()
        sh[d] = float(np.mean([abs(m[(pid, ax)] - b[(pid, ax)]) for pid in m.index.get_level_values(0).unique()]))
        sh[d + "_off"] = float(np.mean([abs(m[(pid, oax)] - b[(pid, oax)]) for pid in m.index.get_level_values(0).unique() for oax in [{"econ": "soc", "soc": "econ"}[ax]]]))
    on = float(np.mean([sh["Ep"], sh["Em"], sh["Sp"], sh["Sm"]])); grid.append(dict(f=f, on_axis_shift=on, **sh)); lc.log(f"f {f}: on-axis shift {on:.3f} | {dict((k, round(v, 3)) for k, v in sh.items())}")
G = pd.DataFrame(grid); hit = G[G.on_axis_shift >= TARGET]
if len(hit): f_H, status = float(hit.f.iloc[0]), "ok"
else:   # A2 fallback: among fractions whose off-axis shift stays below the on-axis shift for every pure push, the smallest one within 0.01 of the largest on-axis shift
    spec = G[[all(r[d + "_off"] < r[d] for d in ("Ep", "Em", "Sp", "Sm")) for _, r in G.iterrows()]]
    spec = spec if len(spec) else G
    f_H = float(spec[spec.on_axis_shift >= spec.on_axis_shift.max() - 0.01].f.iloc[0]); status = f"target not reached; A2 fallback (max on-axis shift {spec.on_axis_shift.max():.3f} among specific fractions)"
if a.model == "qwen": f_H, status = 0.28, "fixed by PLAN (Study K norm); grid for comparison"
out = dict(model=a.model, layer=L, mu=mu, target=TARGET, grid=grid, frac_H=f_H, frac_L=round(f_H / 1.4, 4), status=status, base_p_own=b.round(3).to_dict().__repr__())
lc.atomic_json(out, od / "norms.json"); lc.log("norms:", {k: out[k] for k in ("frac_H", "frac_L", "status", "mu")})
