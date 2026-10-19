#!/usr/bin/env python
"""S4b selection (PLAN addendum A7): real cross-pressured members chosen from roll-call votes only (no model coordinates read).
Writes out/s4b/selection.csv (no map positions) and a blind coding sheet with Wikipedia texts fetched by s4_fetch-like code."""
import numpy as np, pandas as pd
import dcommon as dc

O = dc.OUT / "s4b"; O.mkdir(parents=True, exist_ok=True)
sc = pd.read_csv(dc.ISSUE_SCALES); sc = sc[sc.issue.isin(["economic", "abortion"]) & (sc.congress >= 93)]
w = sc.pivot_table(index=["congress", "chamber", "icpsr"], columns="issue", values="rc_score").reset_index().dropna()
for c in ["economic", "abortion"]:
    g = w.groupby(["congress", "chamber"])[c]
    w[f"z_{c}"] = (w[c] - g.transform("median")) / g.transform("std")
known = None
for k in dc.MODELS:   # recall flag only (party known); coordinates are not read here
    s = pd.read_csv(dc.OUT / k / "s3" / "temporal_member_scores.csv", usecols=["congress", "icpsr", "known", "party", "name", "chamber", "state", "year"])
    s = s.rename(columns={"known": f"known_{k}"})
    known = s if known is None else known.merge(s[["congress", "icpsr", f"known_{k}"]], on=["congress", "icpsr"], validate="one_to_one")
d = w.merge(known, on=["congress", "chamber", "icpsr"], validate="one_to_one")
d = d[d.party.isin(["Democrat", "Republican"]) & d.known_qwen.astype(bool) & d.known_llama.astype(bool)]
m = d.groupby("icpsr").agg(name=("name", "first"), party=("party", lambda x: x.mode().iloc[0]), chamber=("chamber", lambda x: "/".join(sorted(set(x)))),
                           state=("state", lambda x: "/".join(sorted(set(x)))), first=("year", "min"), last=("year", "max"), n=("year", "size"),
                           z_econ=("z_economic", "mean"), z_abort=("z_abortion", "mean")).reset_index()
m = m[m.n >= 2].copy()
m["vote_q"] = pd.Series(np.where(m.z_econ > 0, "ER-", "EL-"), index=m.index) + np.where(m.z_abort > 0, "ST", "SP")
m["depth"] = np.minimum(m.z_econ.abs(), m.z_abort.abs())
sel = pd.concat([m[m.vote_q == q].nlargest(10 if q in ("EL-ST", "ER-SP") else 5, "depth").assign(group="cross" if q in ("EL-ST", "ER-SP") else "aligned_ref")
                 for q in ["EL-ST", "ER-SP", "EL-SP", "ER-ST"]])
vv = pd.read_csv(dc.MEMBERS, low_memory=False)[["icpsr", "bioguide_id"]].dropna().drop_duplicates("icpsr")
sel = sel.merge(vv, on="icpsr", how="left")
dc.atomic_csv(sel, O / "selection.csv")
dc.log(f"S4b: {len(sel)} members; eligible per vote quadrant {m.vote_q.value_counts().to_dict()}")
