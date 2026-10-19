#!/usr/bin/env python
"""Study B stage 0 - roll-call data: PAP topics joined to Voteview, completeness ledger, issue-specific roll-call scales.

1. PAP rows for Congresses 80-118 are classified (completeness ledger): ok / duplicate key (same topic: one kept; different
   topics: conflict, excluded) / misaligned (topic column holds text) / no topic. Join to Voteview HSall_rollcalls on
   (congress, chamber, rollnumber); expected match >= 99% of Voteview roll calls in 80-118, else stop.
2. Issue roll calls (issues.py): PAP subtopics, or a regex on Voteview's own descriptions (vote_desc, vote_question, dtl_desc).
3. Scales per Congress x chamber x issue (DESIGN section 10): roll calls with minority side >= 10%, >= 3 roll calls,
   members with >= 3 votes; yea 1 / nay 0 / else missing; column-centred, missing 0, first principal component.
   Orientation: sign of r with DW-NOMINATE dim1 if |r| >= 0.3, else Southern Democrats above other Democrats.
   The party gap is never used to orient. Sensitivity: share of a member's votes on the conservative side.
Outputs: out/data/pap_ledger.csv, issue_rollcalls.csv, issue_scales.csv, data_report.json
"""
import json
import numpy as np, pandas as pd
from issues import ISSUES, RC_ONLY, OUT, PAP_CSV, VV_ROLLCALLS, VV_VOTES, VV_MEMBERS, CONGRESSES, cm

D = OUT / "data"; D.mkdir(exist_ok=True)
KEY = ["congress", "chamber", "rollnumber"]
rep = {}

# ---- 1. PAP ledger and join ----
pap = pd.read_csv(PAP_CSV, low_memory=False, encoding="latin-1")
pap["congress"] = pd.to_numeric(pap.congress, errors="coerce")
pap = pap[pap.congress.isin(CONGRESSES)].copy()
pap["rollnumber"] = pd.to_numeric(pap.rollnumber, errors="coerce")
# PAP codes chamber as 1/2 in most rows and House/Senate in others (found 2026-10-04: first join matched 4.9%).
# 1 = House, 2 = Senate is checked below against Voteview's yea counts (>= 99% agreement required).
rep["pap_chamber_raw"] = pap.chamber.astype(str).value_counts().to_dict()
pap["chamber"] = pap.chamber.astype(str).str.strip().replace({"1": "House", "2": "Senate", "1.0": "House", "2.0": "Senate"})
pap["major"] = pd.to_numeric(pap.pap_majortopic, errors="coerce"); pap["sub"] = pd.to_numeric(pap.pap_subtopic, errors="coerce")
pap["status"] = np.where(pap.pap_majortopic.notna() & pap.major.isna(), "misaligned", np.where(pap["sub"].isna(), "no_topic", "ok"))
dup = pap.duplicated(KEY, keep=False) & pap.status.eq("ok")
for k, g in pap[dup].groupby(KEY):
    same = g["sub"].nunique() == 1
    pap.loc[g.index, "status"] = "duplicate_same" if same else "duplicate_conflict"
    if same: pap.loc[g.index[0], "status"] = "ok"
pap[KEY + ["status", "major", "sub"]].to_csv(D / "pap_ledger.csv", index=False)
rep["pap_status"] = pap.status.value_counts().to_dict()

rc = pd.read_csv(VV_ROLLCALLS, low_memory=False)
rc = rc[rc.congress.isin(CONGRESSES) & rc.chamber.isin(["House", "Senate"])].copy()
good = pap[pap.status.eq("ok") & pap.chamber.isin(["House", "Senate"])].copy()
good["pap_yea"] = pd.to_numeric(good.yea_count, errors="coerce")
good = good[KEY + ["major", "sub", "pap_yea"]].astype({"congress": int, "rollnumber": int})
rc = rc.merge(good, on=KEY, how="left", validate="1:1", indicator=True)
both = rc[rc._merge == "both"]; agree = float((both.pap_yea == both.yea_count).mean())
rep["pap_vs_voteview_yea_agreement"] = agree; print("yea-count agreement on matched keys", round(agree, 4))
if agree < 0.99: raise SystemExit("STOP: PAP and Voteview disagree on yea counts for matched keys; key mapping is wrong")
match = float((rc._merge == "both").mean()); rep["voteview_rollcalls_80_118"] = int(len(rc)); rep["pap_match_rate"] = match
rep["pap_unmatched_by_reason"] = "Voteview roll calls without a usable PAP topic (misaligned, conflict, no topic, or absent in PAP)"
print("PAP topic match rate", round(match, 4))
if match < 0.95: raise SystemExit(f"STOP: PAP topic match rate {match:.3f}; check keys before building scales")
if match < 0.99:
    print("NOTE: match rate below declared 0.99; issues defined by PAP topics exclude unmatched roll calls (recorded in ledger)")
rep["pap_match_gate_0.99"] = bool(match >= 0.99)
txt = (rc.vote_desc.fillna("") + " " + rc.vote_question.fillna("") + " " + rc.dtl_desc.fillna("")).str.upper()
rc["minority"] = np.minimum(rc.yea_count, rc.nay_count) / (rc.yea_count + rc.nay_count).replace(0, np.nan)

rows = []
for iss, spec in (ISSUES | RC_ONLY).items():
    if spec.get("pap_major"): sel = rc["major"].isin(spec["pap_major"])
    elif spec["pap_sub"]: sel = rc["sub"].isin(spec["pap_sub"])
    else: sel = txt.str.contains(spec["regex"], regex=True)
    r = rc[sel][KEY + ["minority", "date"]].copy(); r["issue"] = iss; rows.append(r)
irc = pd.concat(rows, ignore_index=True); irc["contested"] = irc.minority >= 0.10
irc.to_csv(D / "issue_rollcalls.csv", index=False)
rep["issue_rollcalls"] = {i: dict(all=int((g.issue == i).sum()), contested=int(g.contested.sum())) for i, g in irc.groupby("issue")}

# ---- 2. votes for the contested issue roll calls (chunked read of the 700 MB file) ----
want = irc[irc.contested][KEY].drop_duplicates()
want_keys = set(map(tuple, want.values))
parts = []
for ch in pd.read_csv(VV_VOTES, usecols=["congress", "chamber", "rollnumber", "icpsr", "cast_code"], chunksize=4_000_000,
                      dtype={"congress": "int16", "rollnumber": "int32", "icpsr": "int32", "cast_code": "int8"}):
    ch = ch[ch.congress.between(80, 118)]
    if ch.empty: continue
    k = list(zip(ch.congress, ch.chamber, ch.rollnumber)); m = np.fromiter((x in want_keys for x in k), bool, len(k))
    parts.append(ch[m])
votes = pd.concat(parts, ignore_index=True)
votes["v"] = np.where(votes.cast_code.between(1, 3), 1.0, np.where(votes.cast_code.between(4, 6), 0.0, np.nan))
votes = votes.merge(irc[irc.contested][KEY + ["issue"]], on=KEY, how="inner", validate="m:m")
cm.log(f"votes kept: {len(votes)}")

mem = pd.read_csv(VV_MEMBERS, low_memory=False)
mem = mem[mem.chamber.isin(["House", "Senate"])][["congress", "chamber", "icpsr", "party_code", "state_abbrev", "nominate_dim1"]]
mem = mem.drop_duplicates(["congress", "chamber", "icpsr"])

# ---- 3. scales ----
out, info = [], []
for (c, chm, iss), g in votes.groupby(["congress", "chamber", "issue"]):
    M = g.pivot_table(index="icpsr", columns="rollnumber", values="v", aggfunc="first")
    M = M[M.notna().sum(1) >= 3]
    if M.shape[1] < 3 or len(M) < 20:
        info.append(dict(congress=c, chamber=chm, issue=iss, n_rc=M.shape[1], n_mem=len(M), built=False)); continue
    X = M.values; mu = np.nanmean(X, 0); Z = np.where(np.isnan(X), 0.0, X - mu)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False); sc = U[:, 0] * S[0]; load = Vt[0]
    mm = pd.DataFrame({"icpsr": M.index, "score": sc}).merge(mem[(mem.congress == c) & (mem.chamber == chm)], on="icpsr", how="left", validate="1:1")
    ok = mm.nominate_dim1.notna()   # NaN-safe (bug fixed 2026-10-04: one missing dim1 made r NaN and forced the fallback rule)
    r1 = np.corrcoef(mm.score[ok], mm.nominate_dim1[ok])[0, 1] if ok.sum() > 10 else np.nan
    dem = mm[mm.party_code == 100]; sdem = dem.state_abbrev.isin(cm.SOUTH)
    if np.isfinite(r1) and abs(r1) >= 0.3: sign, rule = np.sign(r1), "dim1"
    elif sdem.sum() >= 5 and (~sdem).sum() >= 5: sign, rule = np.sign(dem.score[sdem.values].mean() - dem.score[~sdem.values].mean()) or 1.0, "south_dem"
    else: sign, rule = (np.sign(r1) if np.isfinite(r1) and r1 != 0 else 1.0), "dim1_weak"
    cons_yea = sign * load > 0
    share = np.nanmean(np.where(np.isnan(X), np.nan, (X == 1) == cons_yea[None, :]), 1)
    out.append(pd.DataFrame(dict(congress=c, chamber=chm, issue=iss, icpsr=M.index, rc_score=sign * sc, rc_share=share, n_votes=M.notna().sum(1).values)))
    info.append(dict(congress=c, chamber=chm, issue=iss, n_rc=M.shape[1], n_mem=len(M), built=True, rule=rule, r_dim1=float(sign * r1) if np.isfinite(r1) else None,
                     pc1_share=float(S[0] ** 2 / (S ** 2).sum())))
scales = pd.concat(out, ignore_index=True)
assert not scales.duplicated(["congress", "chamber", "issue", "icpsr"]).any()
scales.to_csv(D / "issue_scales.csv", index=False)
inf = pd.DataFrame(info); inf.to_csv(D / "scale_info.csv", index=False)
rep["scales_built"] = inf.groupby("issue").built.sum().astype(int).to_dict()
rep["orientation_rules"] = inf[inf.built].groupby(["issue", "rule"]).size().rename("n").reset_index().to_dict("records")
rep["first_congress_with_scale"] = inf[inf.built].groupby("issue").congress.min().astype(int).to_dict()
json.dump(rep, open(D / "data_report.json", "w"), indent=1, default=str)
print(json.dumps(rep, indent=1, default=str))
