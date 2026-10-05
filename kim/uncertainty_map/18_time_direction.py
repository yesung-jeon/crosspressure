#!/usr/bin/env python
"""Step 18 - is time folded into Kim's social axis? Extract a "time direction" and remove it.

1 Time direction. 30 named non-political persons (step 9: name x occupation x state) placed at 1900, 1925, 1950, 1975, 2000,
  2023, 2040, 2060 with the neutral ending. Last-token states at layer 17 on Kim's 10 held-out questions (forward only).
  t = mean over persons 1-15 and questions of h(2023) - h(1950), unit length; persons 16-30 are held out (Kim K7).
  Reported: cosine of t with orth_s and orth_e; held-out projections on t, y and x by year (does "progress" extrapolate to
  2040 and 2060?).
2 Legislator panel. 300 members with >= 4 Congresses (150 D, 150 R; seed), every Congress they served (Kim's persona, same
  questions). Per member-Congress: y and x as in Kim (int8, same run), the time projection, and y_perp = projection on the social
  axis with the time direction removed (orth_s minus its t component, renormalised), each minus the no-persona baseline.
  Saved for 19's dynamics and responsiveness tests; summarised here.
Output: out/time_direction.npy, out/time_persons.csv, out/time_panel.csv, out/TIME_RESULTS.md (summary written by step 19)
Usage: python 18_time_direction.py [--out out] [--quant int8]
"""
import argparse
import numpy as np, pandas as pd
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--quant", default="int8"); args = ap.parse_args()
OUT = cm.HERE / args.out
OCC = ["a nurse", "a software engineer", "an accountant", "an electrician", "a high school teacher", "a dental hygienist",
       "a truck driver", "a pharmacist", "a restaurant manager", "a librarian"]
STATES = ["Ohio", "California", "Georgia"]
NAMES = ["Emily Johnson", "Michael Smith", "Maria Lopez", "David Lee", "Sarah Williams", "James Brown", "Jessica Davis", "Robert Miller",
         "Ashley Wilson", "Daniel Moore", "Jennifer Taylor", "Christopher Anderson", "Amanda Thomas", "Matthew Jackson", "Stephanie White",
         "Joshua Harris", "Laura Martin", "Andrew Thompson", "Rachel Garcia", "Kevin Martinez", "Nicole Robinson", "Brian Clark",
         "Megan Rodriguez", "Jason Lewis", "Lauren Walker", "Ryan Hall", "Hannah Allen", "Justin Young", "Olivia King", "Eric Wright"]
YEARS = [1900, 1925, 1950, 1975, 2000, 2023, 2040, 2060]
tok, model, V = cm.load(args.quant)
qs = cm.kim_questions(); E = [j for j, q in enumerate(qs) if q["axis"] == "economic"]; S = [j for j, q in enumerate(qs) if q["axis"] == "social"]
ve, vs = V["orth_e"].float().numpy(), V["orth_s"].float().numpy()
base = cm.states(tok, model, V, [cm.chat(tok, q["question"]) for q in qs])           # (10, d) no-persona

# ---- 1. persons across years ----
pp = []
for k, (occ, st) in enumerate([(o, s) for o in OCC for s in STATES]):
    for yr in YEARS:
        pp.append(dict(person=k, name=NAMES[k], year=yr, half="extract" if k < 15 else "heldout",
                       system=f"You are {NAMES[k]}, who works as {occ} in {st} in {yr}. Answer the question in the first person, as {NAMES[k]} would."))
P = pd.DataFrame(pp)
H = cm.states(tok, model, V, [cm.chat(tok, q["question"], s) for s in P.system for q in qs]).reshape(len(P), len(qs), -1)
ex = P.half == "extract"
d = H[(ex & (P.year == 2023)).values].mean((0, 1)) - H[(ex & (P.year == 1950)).values].mean((0, 1))
t = d / np.linalg.norm(d); np.save(OUT / "time_direction.npy", t)
P["t_proj"] = (H.mean(1) - base.mean(0)) @ t                                     # + = later
P["y"] = -((H[:, S] @ vs).mean(1) - (base[S] @ vs).mean()); P["x"] = -((H[:, E] @ ve).mean(1) - (base[E] @ ve).mean())
P.drop(columns="system").to_csv(OUT / "time_persons.csv", index=False)
cm.log(f"cos(t, orth_s) = {float(t @ vs):+.3f}; cos(t, orth_e) = {float(t @ ve):+.3f}")

# ---- 2. legislator panel ----
m = cm.members(); m = m[m.party.isin(["Democrat", "Republican"])]
n_c = m.groupby("icpsr").congress.transform("size"); cand = m[n_c >= 4].drop_duplicates("icpsr")
rng = np.random.default_rng(cm.seed_for("step18-panel"))
ids = np.concatenate([g.icpsr.values[rng.choice(len(g), 150, replace=False)] for _, g in cand.groupby("party")])
panel = m[m.icpsr.isin(ids)].copy()
Hm = cm.states(tok, model, V, [cm.chat(tok, q["question"], cm.persona(r["name"], r.chamber, r.state, int(r.year))) for _, r in panel.iterrows() for q in qs]).reshape(len(panel), len(qs), -1)
s_perp = vs - (vs @ t) * t; s_perp /= np.linalg.norm(s_perp)
panel["y_int8"] = -((Hm[:, S] @ vs).mean(1) - (base[S] @ vs).mean()); panel["x_int8"] = -((Hm[:, E] @ ve).mean(1) - (base[E] @ ve).mean())
panel["y_perp"] = -((Hm[:, S] @ s_perp).mean(1) - (base[S] @ s_perp).mean())
panel["t_proj"] = (Hm.mean(1) - base.mean(0)) @ t
panel[["icpsr", "congress", "year", "name", "party", "chamber", "state", "known", "era", "dim1", "dim2", "x", "y", "x_int8", "y_int8", "y_perp", "t_proj"]].to_csv(OUT / "time_panel.csv", index=False)
cm.log(f"panel: {len(panel)} member-Congresses, {panel.icpsr.nunique()} members"); (OUT / "step18.done").touch()
