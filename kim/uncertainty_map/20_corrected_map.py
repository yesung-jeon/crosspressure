#!/usr/bin/env python
"""Step 20 - A: Kim's congress map re-projected with two corrections for the model's present-day lens.

Every member-Congress in Kim's file (21,288 rows, 1947-2023): Kim's persona x his 10 held-out questions, last-token state at
layer 17 projected on five directions: orth_e, orth_s (Kim), t (time direction, step 18), e_perp and s_perp (Kim's axes with t
removed, renormalised). Economic directions are averaged over the 5 economic questions, social over the 5 social questions,
t over all 10 (as in Kim's scores).
Correction 1 (primary): year-matched ordinary people. For every Congress year, the 30 named non-political persons of step 9
(name x occupation x state, neutral ending) are placed in that year and projected the same way; their mean is the origin for
that year. Corrected x_c, y_c = member minus same-year ordinary people (sign as Kim: + = economic right / traditional).
Reading: "how far right / traditional of an ordinary American of the time does the model place this member".
Correction 2 (sensitivity): x_perp, y_perp on the time-removed axes, origin = the 2023 ordinary people.
Raw Kim-style x, y (origin = the no-persona model, int8, same run) are kept for comparison.
One output file per Congress (resumable). Batch 64, forward only, states are never stored.
Output: out/corrected/members_<congress>.csv, out/corrected/ordinary_<congress>.csv
Usage: python 20_corrected_map.py [--out out] [--quant int8]
"""
import argparse
import numpy as np, pandas as pd, torch
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--quant", default="int8")
ap.add_argument("--batch", type=int, default=64); args = ap.parse_args()
OUT = cm.HERE / args.out; CD = OUT / "corrected"; CD.mkdir(parents=True, exist_ok=True)
OCC = ["a nurse", "a software engineer", "an accountant", "an electrician", "a high school teacher", "a dental hygienist",
       "a truck driver", "a pharmacist", "a restaurant manager", "a librarian"]
STATES = ["Ohio", "California", "Georgia"]
NAMES = ["Emily Johnson", "Michael Smith", "Maria Lopez", "David Lee", "Sarah Williams", "James Brown", "Jessica Davis", "Robert Miller",
         "Ashley Wilson", "Daniel Moore", "Jennifer Taylor", "Christopher Anderson", "Amanda Thomas", "Matthew Jackson", "Stephanie White",
         "Joshua Harris", "Laura Martin", "Andrew Thompson", "Rachel Garcia", "Kevin Martinez", "Nicole Robinson", "Brian Clark",
         "Megan Rodriguez", "Jason Lewis", "Lauren Walker", "Ryan Hall", "Hannah Allen", "Justin Young", "Olivia King", "Eric Wright"]
PERSONS = [(NAMES[k], o, s) for k, (o, s) in enumerate([(o, s) for o in OCC for s in STATES])]

tok, model, V = cm.load(args.quant)
ve, vs = V["orth_e"].float().numpy(), V["orth_s"].float().numpy(); t = np.load(OUT / "time_direction.npy")
def perp(v):
    w = v - (v @ t) * t; return w / np.linalg.norm(w)
DIRS = np.stack([ve, vs, t, perp(ve), perp(vs)]); NAMES_D = ["e", "s", "t", "e_perp", "s_perp"]
qs = cm.kim_questions(); E = [j for j, q in enumerate(qs) if q["axis"] == "economic"]; S = [j for j, q in enumerate(qs) if q["axis"] == "social"]
def scores(systems):
    """Kim-style per-persona scores on the five directions (no baseline subtracted)."""
    pr = cm.project_dirs(tok, model, V, [cm.chat(tok, q["question"], s) for s in systems for q in qs], DIRS, batch=args.batch).reshape(len(systems), len(qs), len(DIRS))
    return pd.DataFrame({"e": pr[:, E, 0].mean(1), "s": pr[:, S, 1].mean(1), "t": pr[:, :, 2].mean(1), "e_perp": pr[:, E, 3].mean(1), "s_perp": pr[:, S, 4].mean(1)})
base = scores([None]).iloc[0]                                                     # no-persona model (Kim's origin)
ref23 = CD / "ordinary_2023.csv"
if not ref23.exists():
    o = scores([f"You are {n}, who works as {oc} in {st} in 2023. Answer the question in the first person, as {n} would." for n, oc, st in PERSONS]); o.to_csv(ref23, index=False)
o23 = pd.read_csv(ref23).mean()
m = cm.members()
for c, g in m.groupby("congress"):
    f_m, f_o = CD / f"members_{c}.csv", CD / f"ordinary_{c}.csv"
    if f_m.exists() and f_o.exists(): continue
    yr = cm.year_of(c)
    o = scores([f"You are {n}, who works as {oc} in {st} in {yr}. Answer the question in the first person, as {n} would." for n, oc, st in PERSONS])
    o.assign(congress=c, year=yr).to_csv(f_o, index=False); om = o.mean()
    sc = scores([cm.persona(r["name"], r.chamber, r.state, int(r.year)) for _, r in g.iterrows()])
    out = g[["congress", "year", "icpsr", "name", "party", "chamber", "state", "known", "dim1", "dim2", "x", "y", "era"]].reset_index(drop=True).rename(columns={"x": "x_kim", "y": "y_kim"})
    out["x_raw"], out["y_raw"] = -(sc.e - base.e), -(sc.s - base.s)                          # Kim's definition, int8, this run
    out["x_c"], out["y_c"] = -(sc.e - om.e), -(sc.s - om.s)                                  # correction 1: same-year ordinary people
    out["x_perp"], out["y_perp"] = -(sc.e_perp - o23.e_perp), -(sc.s_perp - o23.s_perp)      # correction 2: time direction removed
    out["t_proj"] = sc.t - base.t
    out.to_csv(f_m, index=False); cm.log(f"Congress {c}: {len(out)} members")
(OUT / "step20.done").touch(); cm.log("step 20 done")
