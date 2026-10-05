#!/usr/bin/env python
"""Step 1 - generation entropy for legislator personas (for ideas A, B, E).

Sample: every other Congress from 80 to 118 (20 Congresses); in each, one member per state (random, either chamber,
known or not; Kim's party-recall flag is kept for subgroup reports). Prompt: Kim's persona (name, chamber, state, year;
no party) as system turn + each of Kim's 10 held-out questions + the notebook's 100-word suffix; 1 sample per question.
Rows already generated with identical settings in kim/2_runs/R1_explore (E5, E8, E10: 1,064 members) are imported first
and marked source = "explore_cache"; only missing member-Congress pairs are generated (source = "step1").

Output: out/member_generations.csv  one row per member x Congress x question
  icpsr, congress, name, party, chamber, state, year, known, era, group, x, y, dim1, dim2, gmp, q, axis,
  entropy64, n_tokens, hedge, text, source, batch_id
Usage: python 01_generate_members.py [--out out] [--quant int8]
"""
import argparse
import numpy as np, pandas as pd
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--quant", default="int8")
ap.add_argument("--batch", type=int, default=96); args = ap.parse_args()
OUT = cm.HERE / args.out; OUT.mkdir(exist_ok=True); dest = OUT / "member_generations.csv"
KEEP = ["icpsr", "congress", "name", "party", "chamber", "state", "year", "known", "era", "group", "x", "y", "dim1", "dim2", "gmp"]
m = cm.members()
qs = cm.kim_questions()

# ---- 1. import cache (identical prompt, model precision, sampling, entropy definition) ----
cache_files = [cm.HERE.parent / "2_runs" / "R1_explore" / f for f in ("E5E8_personas.csv", "E10_personas.csv")]
if not dest.exists() and not any(f.exists() for f in cache_files):     # fresh checkout without the exploration cache: start empty
    pd.DataFrame(columns=KEEP + ["q", "axis", "entropy64", "n_tokens", "hedge", "text", "batch_id", "source"]).to_csv(dest, index=False)
if not dest.exists():
    cache = pd.concat([pd.read_csv(f) for f in cache_files if f.exists()])
    cache = cache.drop_duplicates(["icpsr", "congress", "q"])
    cache["hedge"] = cache.text.map(cm.hedging_density)
    cache = cache[["icpsr", "congress", "q", "axis", "entropy64", "n_tokens", "hedge", "text", "batch_id"]]
    cache = cache.merge(m[KEEP], on=["icpsr", "congress"], how="left", validate="m:1").assign(source="explore_cache")
    assert cache.name.notna().all(), "cache rows must match Kim's member file"
    cache.to_csv(dest, index=False); cm.log(f"imported cache: {cache[['icpsr','congress']].drop_duplicates().shape[0]} member-Congresses")

# ---- 2. sample: one member per state per Congress, every other Congress ----
rng = np.random.default_rng(cm.seed_for("step1-sample"))
pool = m[m.congress.isin(range(80, 119, 2)) & m.state.isin(cm.STATES.keys())]
pick = pool.groupby(["congress", "state"], group_keys=False).apply(lambda g: g.iloc[[rng.integers(len(g))]])
pick.to_csv(OUT / "step1_sample.csv", index=False)
have = pd.read_csv(dest, usecols=["icpsr", "congress"]).drop_duplicates()
todo = pick.merge(have, on=["icpsr", "congress"], how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
cm.log(f"sample {len(pick)} member-Congresses; {len(todo)} need generation")

# ---- 3. generate missing ----
if len(todo):
    tok, model, V = cm.load(args.quant)
    jobs = [{k: r[k] for k in KEEP} | dict(q=qi, axis=q["axis"], source="step1",
            prompt=cm.chat(tok, q["question"] + cm.SUFFIX, cm.persona(r["name"], r.chamber, r.state, int(r.year))))
            for _, r in todo.iterrows() for qi, q in enumerate(qs)]
    tmp = OUT / "_step1_new.csv"
    cm.generate_entropy(tok, model, jobs, tmp, "step1", batch=args.batch)
    new = pd.read_csv(tmp)
    pd.concat([pd.read_csv(dest), new]).to_csv(dest, index=False); tmp.unlink()
d = pd.read_csv(dest)
cm.log(f"saved {dest}: {len(d)} rows, {d[['icpsr','congress']].drop_duplicates().shape[0]} member-Congresses, sources {d.source.value_counts().to_dict()}")
(OUT / "step1.done").touch()   # run_all.sh skips this GPU step when the marker exists
