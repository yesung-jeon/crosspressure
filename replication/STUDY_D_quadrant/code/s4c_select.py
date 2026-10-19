#!/usr/bin/env python
"""S4c selection (PLAN addendum A8): per vote quadrant, 10 deepest + 5 random members, votes only (no model coordinates read).
Same eligibility and vote positions as s4b_select.py. Writes out/s4c/selection.csv and a coding sheet with only the people not
already coded in S4b."""
import hashlib
import numpy as np, pandas as pd
import dcommon as dc

O = dc.OUT / "s4c"; O.mkdir(parents=True, exist_ok=True)
sc = pd.read_csv(dc.ISSUE_SCALES); sc = sc[sc.issue.isin(["economic", "abortion"]) & (sc.congress >= 93)]
w = sc.pivot_table(index=["congress", "chamber", "icpsr"], columns="issue", values="rc_score").reset_index().dropna()
for c in ["economic", "abortion"]:
    g = w.groupby(["congress", "chamber"])[c]
    w[f"z_{c}"] = (w[c] - g.transform("median")) / g.transform("std")
known = None
for k in dc.MODELS:
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
parts = []
for q in ["EL-SP", "EL-ST", "ER-SP", "ER-ST"]:
    h = m[m.vote_q == q]; deep = h.nlargest(10, "depth")
    rest = h[~h.icpsr.isin(deep.icpsr)].sort_values("icpsr")
    rnd = rest.iloc[np.random.default_rng(dc.seed_for("D-S4c", q) % 2**32).choice(len(rest), 5, replace=False)]
    parts += [deep.assign(how="deepest"), rnd.assign(how="random")]
sel = pd.concat(parts)
vv = pd.read_csv(dc.MEMBERS, low_memory=False)[["icpsr", "bioguide_id"]].dropna().drop_duplicates("icpsr")
sel = sel.merge(vv, on="icpsr", how="left")
dc.atomic_csv(sel, O / "selection.csv")
done = set(pd.read_csv(dc.OUT / "s4b" / "coding_sheet.csv").icpsr)
new = sel[~sel.icpsr.isin(done)].copy()
new["years"] = new["first"].astype(str) + "-" + (new["last"] + 1).astype(str)
new["o"] = [hashlib.sha256(f"D-s4c|{i}".encode()).hexdigest() for i in new.icpsr]
sh = new.sort_values("o")[["icpsr", "name", "chamber", "state", "years", "bioguide_id"]].reset_index(drop=True); sh.insert(0, "row", range(1, len(sh) + 1))
dc.atomic_csv(sh, O / "coding_sheet.csv")
dc.log(f"S4c: {len(sel)} selected ({sel.groupby(['vote_q', 'how']).size().to_dict()}); new to code {len(sh)}")
