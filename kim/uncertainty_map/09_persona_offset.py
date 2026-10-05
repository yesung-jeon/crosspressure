#!/usr/bin/env python
"""Step 9 - where is the origin of Kim's social axis? Decompose the shift that any persona produces.

Fact checked first: with no system turn, the Qwen2.5 chat template inserts "You are Qwen, created by Alibaba Cloud. You are
a helpful assistant." So Kim's origin is the assistant identity, and every persona REPLACES it. T1 found that no human
persona, even a strongly progressive one, lands socially progressive of that origin. Which part of a persona moves y?

C1 explicit default system text (sanity: must equal the origin)
C2 "You are a helpful assistant." (assistant without the Qwen identity)
C3 bare humans: "You are a person." / "You are an American." / "You are a human being."
C4 non-political personas: 30 = 10 occupations x 3 states, common first and last names, year 2023, neutral ending
C5 the same 30 with Kim's ending ("... reflecting this person's actual political positions at that time.")
C6 names only: the 30 names, "You are {name}."
C7 the C4 personas in 1955 (time without politics)
Measures: Kim's projection (x, y) and entropy on Kim's 10 held-out questions (2 samples), as in steps 1-7.
Batch 48 (step 7 filled GPU memory at 96 with long system prompts; mistake log M10).

Output: out/offset_personas.csv, out/offset_generations.csv
Usage: python 09_persona_offset.py [--out out] [--quant int8]
"""
import argparse
import pandas as pd
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--quant", default="int8")
ap.add_argument("--batch", type=int, default=48); ap.add_argument("--rollouts", type=int, default=2); args = ap.parse_args()
OUT = cm.HERE / args.out; OUT.mkdir(exist_ok=True)
DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
OCC = ["a nurse", "a software engineer", "an accountant", "an electrician", "a high school teacher", "a dental hygienist",
       "a truck driver", "a pharmacist", "a restaurant manager", "a librarian"]
STATES = ["Ohio", "California", "Georgia"]
NAMES = ["Emily Johnson", "Michael Smith", "Maria Lopez", "David Lee", "Sarah Williams", "James Brown", "Jessica Davis", "Robert Miller",
         "Ashley Wilson", "Daniel Moore", "Jennifer Taylor", "Christopher Anderson", "Amanda Thomas", "Matthew Jackson", "Stephanie White",
         "Joshua Harris", "Laura Martin", "Andrew Thompson", "Rachel Garcia", "Kevin Martinez", "Nicole Robinson", "Brian Clark",
         "Megan Rodriguez", "Jason Lewis", "Lauren Walker", "Ryan Hall", "Hannah Allen", "Justin Young", "Olivia King", "Eric Wright"]
assert len(NAMES) == len(OCC) * len(STATES) == 30
NEUTRAL_END = "Answer the question in the first person, as {n} would."
KIM_END = "Answer the question in the first person, as {n} would, reflecting this person's actual political positions at that time."

rows = [dict(pid="C0_origin", cond="C0_origin", system=None), dict(pid="C1_default_text", cond="C1_default_text", system=DEFAULT),
        dict(pid="C2_assistant", cond="C2_assistant", system="You are a helpful assistant.")]
rows += [dict(pid=f"C3_{i}", cond="C3_bare_human", system=s) for i, s in enumerate(["You are a person.", "You are an American.", "You are a human being."])]
k = 0
for occ in OCC:
    for st in STATES:
        n = NAMES[k]; k += 1
        rows += [dict(pid=f"C4_{k:02d}", cond="C4_nonpolitical", name=n, system=f"You are {n}, who works as {occ} in {st} in 2023. " + NEUTRAL_END.format(n=n)),
                 dict(pid=f"C5_{k:02d}", cond="C5_nonpolitical_kim_ending", name=n, system=f"You are {n}, who works as {occ} in {st} in 2023. " + KIM_END.format(n=n)),
                 dict(pid=f"C6_{k:02d}", cond="C6_name_only", name=n, system=f"You are {n}."),
                 dict(pid=f"C7_{k:02d}", cond="C7_nonpolitical_1955", name=n, system=f"You are {n}, who works as {occ} in {st} in 1955. " + NEUTRAL_END.format(n=n))]
P = pd.DataFrame(rows); assert P.pid.is_unique

tok, model, V = cm.load(args.quant)
pers_path, gen_path = OUT / "offset_personas.csv", OUT / "offset_generations.csv"
if not pers_path.exists():
    P["x"], P["y"] = cm.persona_scores(tok, model, V, [s if isinstance(s, str) else None for s in P.system])
    P.to_csv(pers_path, index=False); cm.log(f"projections: {len(P)} personas")
P = pd.read_csv(pers_path); qs = cm.kim_questions()
jobs = [dict(pid=r.pid, cond=r.cond, q=qi, axis=q["axis"], rollout=j,
             prompt=cm.chat(tok, q["question"] + cm.SUFFIX, r.system if isinstance(r.system, str) else None))
        for _, r in P.iterrows() for qi, q in enumerate(qs) for j in range(args.rollouts)]
cm.generate_entropy(tok, model, jobs, gen_path, "step9", batch=args.batch)
cm.log("saved", gen_path)
(OUT / "step9.done").touch()   # run_all.sh skips this GPU step when the marker exists
