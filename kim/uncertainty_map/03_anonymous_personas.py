#!/usr/bin/env python
"""Step 3 - the model's political geography without names (idea D).

Persona = Kim's wording with the name removed: "You are a member of the U.S. {chamber} representing {state} in {year}."
Cells: 50 states x 2 chambers x 6 Congresses (82, 90, 98, 106, 114, 118 = 1951, 1967, 1983, 1999, 2015, 2023).
For each cell: Kim's projection (x, y) of the anonymous persona and generation entropy on Kim's 10 held-out questions.
For comparison, every real member of those Congresses is projected in the same run and precision (no generation),
so anonymous and real positions share one measurement.

Output: out/anonymous_cells.csv        (state, chamber, congress, x_anon, y_anon)
        out/anonymous_generations.csv  (cell x question: entropy64, n_tokens, hedge, text)
        out/real_member_projections.csv (icpsr, congress, chamber, state, party, dim1, dim2, x_int8, y_int8)
Usage: python 03_anonymous_personas.py [--out out] [--quant int8]
"""
import argparse
import pandas as pd
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--quant", default="int8")
ap.add_argument("--batch", type=int, default=96); args = ap.parse_args()
OUT = cm.HERE / args.out; OUT.mkdir(exist_ok=True)
CONGRESSES = [82, 90, 98, 106, 114, 118]
STATES50 = [s for s in cm.STATES if s not in ("DC", "PR", "VI", "GU", "AS", "MP")]
assert len(STATES50) == 50
tok, model, V = cm.load(args.quant)

cells_path = OUT / "anonymous_cells.csv"
if not cells_path.exists():
    cells = pd.DataFrame([dict(state=s, chamber=ch, congress=c, year=cm.year_of(c)) for c in CONGRESSES for ch in ("House", "Senate") for s in STATES50])
    cells["x_anon"], cells["y_anon"] = cm.persona_scores(tok, model, V, [cm.anonymous_persona(r.chamber, r.state, r.year) for _, r in cells.iterrows()])
    cells.to_csv(cells_path, index=False); cm.log(f"anonymous projections: {len(cells)} cells")
real_path = OUT / "real_member_projections.csv"
if not real_path.exists():
    m = cm.members(); m = m[m.congress.isin(CONGRESSES)].copy()
    m["x_int8"], m["y_int8"] = cm.persona_scores(tok, model, V, [cm.persona(r["name"], r.chamber, r.state, int(r.year)) for _, r in m.iterrows()])
    m[["icpsr", "congress", "name", "chamber", "state", "party", "known", "dim1", "dim2", "x", "y", "x_int8", "y_int8"]].to_csv(real_path, index=False)
    cm.log(f"real member projections: {len(m)}")
cells = pd.read_csv(cells_path); qs = cm.kim_questions()
jobs = [dict(state=r.state, chamber=r.chamber, congress=int(r.congress), q=qi, axis=q["axis"],
             prompt=cm.chat(tok, q["question"] + cm.SUFFIX, cm.anonymous_persona(r.chamber, r.state, int(r.year))))
        for _, r in cells.iterrows() for qi, q in enumerate(qs)]
cm.generate_entropy(tok, model, jobs, OUT / "anonymous_generations.csv", "step3", batch=args.batch)
cm.log("saved step 3")
(OUT / "step3.done").touch()   # run_all.sh skips this GPU step when the marker exists
