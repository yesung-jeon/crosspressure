#!/usr/bin/env python
"""Step 13 - R4: can the persona predict the member's real roll-call votes, and is it worse where the model hesitates?

Members: every member with persona generations (step 1) in Congresses 100, 106, 112, 118 (1987, 1999, 2011, 2023), both chambers.
Roll calls: per chamber x Congress, 30 contested substantive votes (passage or adoption questions; minority side >= 10% of
votes cast), sampled with a fixed seed. Description: Voteview dtl_desc when informative (older Congresses), otherwise
"{vote_question}: {vote_desc}" (newer Congresses).
Prompt: Kim's persona (system) + "On {date}, the U.S. {chamber} voted on the following question. {description} How did you
vote? Answer with one word: Yea or Nay." Forced choice by next-token probability (no generation; Kim K16 / P2 judge style):
p_yea = P("Y") / (P("Y") + P("N")) for the first tokens of "Yea" / "Nay"; P("Y") + P("N") is kept as format compliance.
Truth: Voteview cast_code 1-3 = yea, 4-6 = nay (others dropped). Baselines: the member's party majority on that roll call,
and DW-NOMINATE's probability of the observed vote (Voteview `prob`).

Output: out/vote_items.csv (roll calls used), out/vote_predictions.csv (member x roll call)
Usage: python 13_vote_prediction.py [--out out] [--quant int8]
"""
import argparse
import numpy as np, pandas as pd, torch
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--quant", default="int8")
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--n_rc", type=int, default=30); args = ap.parse_args()
OUT = cm.HERE / args.out; VV = cm.HERE.parent / "data" / "voteview"
CONG = [100, 106, 112, 118]; CH = {"H": "House", "S": "Senate"}

# ---- roll calls ----
items = []
for c in CONG:
    for k, ch in CH.items():
        rc = pd.read_csv(VV / f"{k}{c}_rollcalls.csv")
        q = rc.vote_question.fillna("").str.lower(); dd = rc.dtl_desc.fillna("")
        subst = q.str.contains("passage|adopt|agree|suspend the rules and pass|cloture") | dd.str.upper().str.match(r"^TO (PASS|ADOPT|AGREE)")
        tot = rc.yea_count + rc.nay_count; contested = np.minimum(rc.yea_count, rc.nay_count) / tot.clip(lower=1) >= 0.10
        cand = rc[subst & contested & (tot > 0)].copy()
        cand["desc"] = np.where(dd.loc[cand.index].str.len() >= 40, dd.loc[cand.index].str.strip(),
                                (rc.vote_question.fillna("") + ": " + rc.vote_desc.fillna(rc.bill_number.fillna(""))).loc[cand.index].str.strip())
        cand = cand[cand.desc.str.len() >= 25]
        rng = np.random.default_rng(cm.seed_for("step13-rc", c, k))
        pick = cand.iloc[np.sort(rng.choice(len(cand), min(args.n_rc, len(cand)), replace=False))]
        items.append(pick[["congress", "chamber", "rollnumber", "date", "yea_count", "nay_count", "vote_question", "desc"]])
items = pd.concat(items); items.to_csv(OUT / "vote_items.csv", index=False); cm.log(f"roll calls: {len(items)}")

# ---- members and truth ----
M = pd.read_csv(OUT / "member_generations.csv", usecols=["icpsr", "congress", "name", "party", "chamber", "state", "year"]).drop_duplicates(["icpsr", "congress"])
M = M[M.congress.isin(CONG) & M.party.isin(["Democrat", "Republican"])]
votes = pd.concat([pd.read_csv(VV / f"{k}{c}_votes.csv") for c in CONG for k in CH])
votes = votes[votes.cast_code.between(1, 6)].assign(yea=lambda d: (d.cast_code <= 3).astype(int))
party = M[["icpsr", "congress", "party"]]
pv = votes.merge(cm.members()[["icpsr", "congress", "party"]], on=["icpsr", "congress"], how="inner", validate="m:1")
pmaj = pv.groupby(["congress", "chamber", "rollnumber", "party"]).yea.mean().rename("party_yea_share").reset_index()
X = items.merge(M, on=["congress", "chamber"], validate="m:m").merge(votes[["congress", "chamber", "rollnumber", "icpsr", "yea", "prob"]],
        on=["congress", "chamber", "rollnumber", "icpsr"], how="inner", validate="1:1")
X = X.merge(pmaj, on=["congress", "chamber", "rollnumber", "party"], how="left", validate="m:1")
cm.log(f"member x roll call pairs with a recorded yea/nay: {len(X)} ({X.icpsr.nunique()} members)")

# ---- forced choice ----
tok, model, _ = cm.load(args.quant)
yid, nid = tok.encode("Yea", add_special_tokens=False)[0], tok.encode("Nay", add_special_tokens=False)[0]
assert yid != nid
def prompt(r):
    sysm = cm.persona(r["name"], r.chamber, r.state, int(r.year))
    u = (f"On {r.date}, the U.S. {cm.CHAMBER_LONG[r.chamber]} voted on the following question.\n{r.desc}\n"
         f"How did you vote? Answer with one word: Yea or Nay.")
    return cm.chat(tok, u, sysm)
P = [prompt(r) for _, r in X.iterrows()]
lens = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in P]); order = np.argsort(lens)
py, mass = np.zeros(len(P)), np.zeros(len(P))
with torch.inference_mode():
    for b0 in range(0, len(P), args.batch):
        idx = order[b0:b0 + args.batch]
        enc = tok([P[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), -1)
        a, b = lp[:, yid].exp(), lp[:, nid].exp()
        py[idx] = (a / (a + b)).cpu().numpy(); mass[idx] = (a + b).cpu().numpy()
        if (b0 // args.batch) % 40 == 0: cm.log(f"  {b0}/{len(P)}")
X["p_yea"], X["format_mass"] = py, mass
X.drop(columns=["desc"]).to_csv(OUT / "vote_predictions.csv", index=False)
cm.log("saved vote predictions")
(OUT / "step13.done").touch()
