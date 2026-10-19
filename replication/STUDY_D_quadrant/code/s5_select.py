#!/usr/bin/env python
"""S5 I4 member selection (PLAN.md, CPU): vote-based quadrant from the economic and abortion roll-call scales (93rd Congress on;
above/below the Congress x chamber median; economic high = right, abortion high = traditional, as E40 V1), D/R members recalled
in BOTH models' S3; 30 members per profile, seeded; one member-Congress row per member (seeded). -> out/s5/i4_members.csv"""
import numpy as np, pandas as pd
import dcommon as dc

O = dc.OUT / "s5"; O.mkdir(parents=True, exist_ok=True)
sc = pd.read_csv(dc.ISSUE_SCALES); sc = sc[sc.issue.isin(["economic", "abortion"]) & (sc.congress >= 93)]
w = sc.pivot_table(index=["congress", "chamber", "icpsr"], columns="issue", values="rc_score").reset_index().dropna()
s = {k: pd.read_csv(dc.OUT / k / "s3" / "temporal_member_scores.csv") for k in dc.MODELS}
base = s["qwen"][["congress", "icpsr", "name", "party", "chamber", "state", "year", "known"]].rename(columns={"known": "known_qwen"})
base = base.merge(s["llama"][["congress", "icpsr", "known"]].rename(columns={"known": "known_llama"}), on=["congress", "icpsr"], validate="one_to_one")
d = base.merge(w, on=["congress", "chamber", "icpsr"], how="inner", validate="one_to_one")
# A15 (2026-10-16): medians on the S4 rule's population (mapped D/R rows, 93rd on, both scales), as PLAN section 5 says; the sealed
# selector took medians over all scale rows before this restriction (249 of 10,444 rows got a different vote quadrant).
d = d[d.party.isin(["Democrat", "Republican"])].copy()
for c in ["economic", "abortion"]:
    d[f"hi_{c}"] = d[c] > d.groupby(["congress", "chamber"])[c].transform("median")
d["vote_q"] = pd.Series(np.where(d.hi_economic, "ER-", "EL-"), index=d.index) + np.where(d.hi_abortion, "ST", "SP")
dc.log(f"A15 median population: {len(d)} mapped D/R rows")
d = d[d.known_qwen.astype(bool) & d.known_llama.astype(bool)]
rows = []
for prof in ["EL-SP", "ER-ST", "EL-ST", "ER-SP"]:
    g = d[d.vote_q == prof]
    rng = np.random.default_rng(dc.seed_for("D-S5-I4", prof) % 2**32)
    people = np.sort(g.icpsr.unique()); pick = rng.choice(people, size=min(30, len(people)), replace=False)
    for p in pick:
        gg = g[g.icpsr == p]; rows.append(gg.iloc[int(rng.integers(len(gg)))].to_dict() | {"profile": prof})
    dc.log(f"{prof}: eligible members {len(people)}, picked {len(pick)}; parties {g[g.icpsr.isin(pick)].drop_duplicates('icpsr').party.value_counts().to_dict()}")
dc.atomic_csv(pd.DataFrame(rows), O / "i4_members.csv")
