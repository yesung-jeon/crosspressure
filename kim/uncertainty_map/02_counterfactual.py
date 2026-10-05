#!/usr/bin/env python
"""Step 2 - same person, different conditions (idea C).

(a) Year swap. 60 members the model knows (Kim's party-recall flag), 10 Democrats + 10 Republicans per era
    (1947-79, 1981-2005, 2007-23). Kim's persona with the year set to 1955, 1985, 2015, and to the member's own year.
    Name, chamber and state are held fixed, so a change is attributable to the year in the prompt.
(b) Party switchers. Members whose Voteview party code changed between 100 and 200 (keyed by bioguide_id, because
    Voteview may assign a new ICPSR id after a switch). Persona at the last Congress under the old party and the first
    under the new party (Kim's wording; the party is never in the prompt).
For every persona: Kim's projection (x, y; no generation) and generation entropy on Kim's 10 held-out questions.

Output: out/counterfactual_personas.csv (one row per persona: design, x, y)
        out/counterfactual_generations.csv (one row per persona x question)
Usage: python 02_counterfactual.py [--out out] [--quant int8]
"""
import argparse
import numpy as np, pandas as pd
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--quant", default="int8")
ap.add_argument("--batch", type=int, default=96); args = ap.parse_args()
OUT = cm.HERE / args.out; OUT.mkdir(exist_ok=True)
m = cm.members(); rng = np.random.default_rng(cm.seed_for("step2-sample"))

# ---- (a) year swap ----
k = m[m.known.astype(bool) & m.party.isin(["Democrat", "Republican"])].drop_duplicates("icpsr")
ys = pd.concat([g.iloc[rng.choice(len(g), 10, replace=False)] for _, g in k.groupby(["era", "party"])])
rows = []
for _, r in ys.iterrows():
    for cond, yr in [("own", int(r.year)), ("y1955", 1955), ("y1985", 1985), ("y2015", 2015)]:
        rows.append(dict(design="year_swap", pid=f"ys_{r.icpsr}", icpsr=int(r.icpsr), name=r["name"], party=r.party, chamber=r.chamber,
                         state=r.state, own_year=int(r.year), era=r.era, cond=cond, prompt_year=yr, dim1=r.dim1, dim2=r.dim2))

# ---- (b) party switchers ----
v = pd.read_csv(cm.VOTEVIEW, low_memory=False)
v = v[v.chamber.isin(["House", "Senate"]) & v.congress.between(80, 118) & v.party_code.isin([100, 200]) & v.bioguide_id.notna()]
sw = v.groupby("bioguide_id").party_code.nunique(); sw = sw[sw > 1].index
n_sw = 0
for bid in sw:
    h = v[v.bioguide_id == bid].sort_values("congress"); pc = h.party_code.values
    j = int(np.argmax(pc != pc[0]))                         # first Congress under the new party
    for cond, r in [("before", h.iloc[j - 1]), ("after", h.iloc[j])]:
        rows.append(dict(design="switcher", pid=f"sw_{bid}", icpsr=int(r.icpsr), name=cm.nice(r.bioname), party={100: "Democrat", 200: "Republican"}[r.party_code],
                         chamber=r.chamber, state=r.state_abbrev, own_year=cm.year_of(r.congress), era="", cond=cond,
                         prompt_year=cm.year_of(r.congress), dim1=r.nominate_dim1, dim2=r.nominate_dim2,
                         switch=f"{ {100: 'D', 200: 'R'}[pc[0]] }->{ {100: 'D', 200: 'R'}[pc[j]] }".replace(" ", "")))
    n_sw += 1
P = pd.DataFrame(rows); cm.log(f"year-swap personas {int((P.design == 'year_swap').sum())} (members {ys.shape[0]}); switchers {n_sw}")
assert (P.design == "switcher").sum() == 2 * n_sw

pers_path, gen_path = OUT / "counterfactual_personas.csv", OUT / "counterfactual_generations.csv"
tok, model, V = cm.load(args.quant)
if not pers_path.exists():
    systems = [cm.persona(r["name"], r.chamber, r.state, int(r.prompt_year)) for _, r in P.iterrows()]
    P["x"], P["y"] = cm.persona_scores(tok, model, V, systems)
    P.to_csv(pers_path, index=False); cm.log("projections saved")
P = pd.read_csv(pers_path); qs = cm.kim_questions()
jobs = [r.drop(["x", "y"]).to_dict() | dict(x=r.x, y=r.y, q=qi, axis=q["axis"],
        prompt=cm.chat(tok, q["question"] + cm.SUFFIX, cm.persona(r["name"], r.chamber, r.state, int(r.prompt_year))))
        for _, r in P.iterrows() for qi, q in enumerate(qs)]
cm.generate_entropy(tok, model, jobs, gen_path, "step2", batch=args.batch)
cm.log("saved", gen_path)
(OUT / "step2.done").touch()   # run_all.sh skips this GPU step when the marker exists
