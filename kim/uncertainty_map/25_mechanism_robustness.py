#!/usr/bin/env python
"""Step 25 - mechanism and robustness runs for the temporal distortion (protocol: kim/0_protocol/PROTOCOL_R3_mechanism_robustness.md).

Generator: Qwen2.5-7B-Instruct int8. Judge: Qwen2.5-3B-Instruct bf16 (blind to persona and year).
  m1   time steering along the time direction (held-out persons 15-29 and the 60 year-swap legislators), generated answers
  m2   year expressions (numeric, decade, era, relative, employee-number control), internal projection
  r1   the paper's 48 WVS questions: year swap, ordinary persons 1955 vs 2023, calendar drift in a 100-member panel, projection
  r3   eight baseline groups of ordinary people x every Congress year, projection
  m4v  validation answers for the placebo judge dimensions (tone, formality)
  judge  social / economic judge on m1 answers; placebo judge on m4v and on existing year-swap and ordinary-person answers
Each stage writes to out/r3/ and is skipped when its output exists.
Usage: python 25_mechanism_robustness.py [--out out] [--stages m1 m2 r1 r3 m4v judge]
"""
import argparse, json
import numpy as np, pandas as pd, torch
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--stages", nargs="+", default=["m1", "m2", "r1", "r3", "m4v", "judge"]); args = ap.parse_args()
OUT = cm.HERE / args.out; R3 = OUT / "r3"; R3.mkdir(parents=True, exist_ok=True)
H_NORM, D_TIME = 70.80492401123047, 13.28          # scale_check.json; held-out t_proj(2023) - t_proj(1950), step 18
A1 = D_TIME / H_NORM
OCC = ["a nurse", "a software engineer", "an accountant", "an electrician", "a high school teacher", "a dental hygienist",
       "a truck driver", "a pharmacist", "a restaurant manager", "a librarian"]
NAMES = ["Emily Johnson", "Michael Smith", "Maria Lopez", "David Lee", "Sarah Williams", "James Brown", "Jessica Davis", "Robert Miller",
         "Ashley Wilson", "Daniel Moore", "Jennifer Taylor", "Christopher Anderson", "Amanda Thomas", "Matthew Jackson", "Stephanie White",
         "Joshua Harris", "Laura Martin", "Andrew Thompson", "Rachel Garcia", "Kevin Martinez", "Nicole Robinson", "Brian Clark",
         "Megan Rodriguez", "Jason Lewis", "Lauren Walker", "Ryan Hall", "Hannah Allen", "Justin Young", "Olivia King", "Eric Wright"]
PERSONS = [(NAMES[k], o, s) for k, (o, s) in enumerate([(o, s) for o in OCC for s in ["Ohio", "California", "Georgia"]])]   # step 9 / 18 order
HELD = list(range(15, 30))
def ordinary(n, o, s, when, extra=""):
    return f"You are {n}{extra}, who works as {o} in {s} {when}. Answer the question in the first person, as {n} would."
qs = cm.kim_questions(); QB = json.load(open(cm.HERE.parent / "1_data_question_bank.json", encoding="utf-8"))
need_gen = any(s in args.stages for s in ("m1", "m2", "r1", "r3", "m4v"))
if need_gen:
    tok, model, V = cm.load("int8"); L = int(V["layer"])
    ve, vs = V["orth_e"].float().numpy(), V["orth_s"].float().numpy(); t = np.load(OUT / "time_direction.npy")
cp = pd.read_csv(OUT / "counterfactual_personas.csv"); legs = cp[(cp.design == "year_swap") & (cp.cond == "own")].reset_index(drop=True)

# ---------------- M1 time steering ----------------
if "m1" in args.stages and not (R3 / "m1_generations.csv").exists():
    g = np.random.default_rng(cm.seed_for("m1-random")); rnd = [cm_u / np.linalg.norm(cm_u) for cm_u in (g.standard_normal(t.size), g.standard_normal(t.size))]
    conds_2023 = [("none", 0 * t), ("past_0.5", -0.5 * A1 * t), ("past_1.0", -A1 * t), ("future_0.5", 0.5 * A1 * t), ("random1_1.0", A1 * rnd[0]), ("random2_1.0", A1 * rnd[1])]
    conds_1955 = [("none", 0 * t), ("present_1.0", A1 * t), ("random1_1.0", A1 * rnd[0])]
    conds_leg = [("none", 0 * t), ("past_1.0", -A1 * t), ("present_0.5", 0.5 * A1 * t), ("random1_1.0", A1 * rnd[0])]
    jobs = []
    for k in HELD:
        n, o, s = PERSONS[k]
        for base_year, conds in ((2023, conds_2023), (1955, conds_1955)):
            for cond, d in conds:
                for qi, q in enumerate(qs):
                    for r in range(2):
                        jobs.append(dict(kind="ordinary", pid=f"o{k}", base_year=base_year, cond=cond, q=qi, axis=q["axis"], rollout=r, steer=d.astype(np.float32),
                                         prompt=cm.chat(tok, q["question"] + cm.SUFFIX, ordinary(n, o, s, f"in {base_year}"))))
    for _, rw in legs.iterrows():
        for cond, d in conds_leg:
            for qi, q in enumerate(qs):
                jobs.append(dict(kind="legislator", pid=rw.pid, base_year=int(rw.own_year), cond=cond, q=qi, axis=q["axis"], rollout=0, steer=d.astype(np.float32),
                                 prompt=cm.chat(tok, q["question"] + cm.SUFFIX, cm.persona(rw["name"], rw.chamber, rw.state, int(rw.own_year)))))
    cm.generate_entropy(tok, model, jobs, R3 / "m1_generations.csv", "step25-m1", batch=args.batch, steer_layer=L)

# ---------------- M2 year expressions ----------------
if "m2" in args.stages and not (R3 / "m2_projections.csv").exists():
    EXPR = {"numeric": ("in 1955", "in 2015"), "decade": ("in the 1950s", "in the 2010s"),
            "era": ("during the Eisenhower administration", "during the Obama administration"), "relative": ("seventy years ago", "ten years ago")}
    rows, systems = [], []
    for k, (n, o, s) in enumerate(PERSONS):
        for ex, (a, b) in EXPR.items():
            for per, w in (("1950s", a), ("2010s", b)):
                rows.append(dict(kind="ordinary", pid=f"o{k}", expr=ex, period=per)); systems.append(ordinary(n, o, s, w))
        for per, num in (("1950s", "1955"), ("2010s", "2015")):
            rows.append(dict(kind="ordinary", pid=f"o{k}", expr="employee_control", period=per)); systems.append(ordinary(n, o, s, "in 2023", f", employee number {num}"))
    for _, rw in legs.head(30).iterrows():
        ch = cm.CHAMBER_LONG[rw.chamber]; st = cm.STATES.get(rw.state, rw.state)
        for ex, (a, b) in EXPR.items():
            for per, w in (("1950s", a), ("2010s", b)):
                rows.append(dict(kind="legislator", pid=rw.pid, expr=ex, period=per))
                systems.append(f"You are {rw['name']}, who serves in the U.S. {ch} representing {st} {w}. Answer the question in the first person, "
                               f"as {rw['name']} would, reflecting this person's actual political positions at that time.")
    P = pd.DataFrame(rows); P["x"], P["y"] = cm.persona_scores(tok, model, V, systems); P.to_csv(R3 / "m2_projections.csv", index=False); cm.log("m2 saved")

# ---------------- R1 WVS questions ----------------
def wvs_scores(systems):
    base = cm.project(tok, model, V, [cm.chat(tok, q["question"]) for q in QB])
    P = cm.project(tok, model, V, [cm.chat(tok, q["question"], s) for s in systems for q in QB]).reshape(len(systems), len(QB), 2)
    return -(P[:, :, 0].mean(1) - base[:, 0].mean()), -(P[:, :, 1].mean(1) - base[:, 1].mean())
if "r1" in args.stages and not (R3 / "r1_projections.csv").exists():
    rows, systems = [], []
    ys = cp[cp.design == "year_swap"]
    for _, rw in ys.iterrows():
        rows.append(dict(part="yearswap", pid=rw.pid, cond=rw.cond, icpsr=rw.icpsr)); systems.append(cm.persona(rw["name"], rw.chamber, rw.state, int(rw.prompt_year)))
    for k, (n, o, s) in enumerate(PERSONS):
        for y in (1955, 2023):
            rows.append(dict(part="ordinary", pid=f"o{k}", cond=f"y{y}")); systems.append(ordinary(n, o, s, f"in {y}"))
    tp = pd.read_csv(OUT / "time_panel.csv"); ids = np.random.default_rng(cm.seed_for("r1-panel")).choice(tp.icpsr.unique(), 100, replace=False)
    for _, rw in tp[tp.icpsr.isin(ids)].iterrows():
        rows.append(dict(part="panel", pid=f"p{rw.icpsr}", cond=str(rw.congress), icpsr=rw.icpsr, congress=rw.congress, year=rw.year, party=rw.party))
        systems.append(cm.persona(rw["name"], rw.chamber, rw.state, int(rw.year)))
    P = pd.DataFrame(rows); P["x_wvs"], P["y_wvs"] = wvs_scores(systems); P.to_csv(R3 / "r1_projections.csv", index=False); cm.log(f"r1 saved: {len(P)}")

# ---------------- R3 baseline groups ----------------
GROUPS = {
    "default": PERSONS[:10],
    "women": [(n, o, s) for n, o, s in zip(["Emily Johnson", "Maria Lopez", "Sarah Williams", "Jessica Davis", "Ashley Wilson", "Jennifer Taylor", "Amanda Thomas", "Stephanie White", "Laura Martin", "Rachel Garcia"], OCC, ["Ohio", "California", "Georgia"] * 4)],
    "men": [(n, o, s) for n, o, s in zip(["Michael Smith", "David Lee", "James Brown", "Robert Miller", "Daniel Moore", "Christopher Anderson", "Matthew Jackson", "Joshua Harris", "Andrew Thompson", "Kevin Martinez"], OCC, ["Ohio", "California", "Georgia"] * 4)],
    "black_names": [(n, o, s) for n, o, s in zip(["Lakisha Washington", "Jamal Jackson", "Tyrone Robinson", "Aisha Williams", "DeShawn Harris", "Latoya Jefferson", "Darnell Brooks", "Keisha Coleman", "Tamika Banks", "Terrell Gaines"], OCC, ["Ohio", "California", "Georgia"] * 4)],
    "hispanic_names": [(n, o, s) for n, o, s in zip(["Jose Hernandez", "Maria Gonzalez", "Luis Ramirez", "Carmen Lopez", "Juan Martinez", "Rosa Torres", "Carlos Rivera", "Ana Flores", "Miguel Sanchez", "Elena Morales"], OCC, ["Ohio", "California", "Georgia"] * 4)],
    "south_rural": [(n, o, s) for n, o, s in zip(NAMES[10:20], ["a farmer", "a mechanic", "a truck driver", "a welder", "a cashier", "a farmer", "a mechanic", "a construction worker", "a cook", "a carpenter"], ["Alabama", "Mississippi", "Georgia"] * 4)],
    "northeast_professional": [(n, o, s) for n, o, s in zip(NAMES[20:30], ["a lawyer", "a professor", "a physician", "an architect", "a journalist", "a lawyer", "a professor", "a consultant", "an editor", "a financial analyst"], ["Massachusetts", "New York", "Connecticut"] * 4)],
    "age70": [(n + ", 70 years old,", "a retired " + o.split(" ", 1)[1], s) for n, o, s in PERSONS[10:20]],
}
if "r3" in args.stages and not (R3 / "r3_baselines.csv").exists():
    years = sorted(cm.members().year.unique()); rows, systems = [], []
    for gname, ppl in GROUPS.items():
        for i, (n, o, s) in enumerate(ppl):
            nm = n.split(",")[0]
            for y in years:
                rows.append(dict(group=gname, person=i, year=int(y)))
                systems.append(f"You are {n} who works as {o} in {s} in {y}. Answer the question in the first person, as {nm} would." if "," in n else ordinary(n, o, s, f"in {y}"))
    P = pd.DataFrame(rows); P["x"], P["y"] = cm.persona_scores(tok, model, V, systems); P.to_csv(R3 / "r3_baselines.csv", index=False); cm.log(f"r3 saved: {len(P)}")

# ---------------- M4 validation answers ----------------
PLACEBO_INSTR = {"tone": ("Answer in a warm, positive and optimistic tone.", "Answer in a cold, negative and pessimistic tone."),
                 "formality": ("Answer in very casual, conversational everyday language.", "Answer in very formal, elevated and ceremonious language.")}
if "m4v" in args.stages and not (R3 / "m4_validation_texts.csv").exists():
    jobs = [dict(dim=dim, q=qi, axis=q["axis"], side=side, prompt=cm.chat(tok, q["question"] + cm.SUFFIX, txt))
            for dim, (pa, pb) in PLACEBO_INSTR.items() for qi, q in enumerate(qs) for side, txt in (("A", pa), ("B", pb)) for _ in range(2)]
    cm.generate_entropy(tok, model, jobs, R3 / "m4_validation_texts.csv", "step25-m4v", batch=args.batch)
if need_gen:
    del model; torch.cuda.empty_cache()

# ---------------- judge ----------------
if "judge" in args.stages:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    jt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct"); jt.padding_side = "left"
    jm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.bfloat16, device_map={"": 0}).eval()
    AID, BID = jt.encode("A", add_special_tokens=False)[0], jt.encode("B", add_special_tokens=False)[0]
    OPT = {"economic": ("more government involvement, public provision or redistribution", "more reliance on markets, private provision or individual responsibility"),
           "social": ("progressive positions that change or relax traditional norms", "traditional positions that preserve established norms"),
           "tone": ("a warm, positive or optimistic tone", "a cold, negative or pessimistic tone"),
           "formality": ("casual, conversational everyday language", "formal, elevated or ceremonious language")}
    QUEST = {"economic": "Which position does this answer take on the question?", "social": "Which position does this answer take on the question?",
             "tone": "Which best describes the tone of this answer?", "formality": "Which best describes the language of this answer?"}
    def jp(q, a, dim):
        oa, ob = OPT[dim]
        u = f"Question: {qs[q]['question']}\n\nAnswer: {str(a).strip()}\n\n{QUEST[dim]}\nA. {oa}\nB. {ob}\nReply with one letter: A or B."
        return jt.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    @torch.inference_mode()
    def judge(df, dim_of, name):
        f = R3 / f"judged_{name}.csv"
        if f.exists(): return
        P = [jp(q, a, dim_of(r)) for (q, a), (_, r) in zip(zip(df.q, df.text), df.iterrows())]
        lens = np.array([len(jt(p, add_special_tokens=False)["input_ids"]) for p in P]); order = np.argsort(lens); sb, mass = np.zeros(len(P)), np.zeros(len(P))
        for b0 in range(0, len(P), 32):
            idx = order[b0:b0 + 32]; enc = jt([P[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(jm.device)
            pr = torch.softmax(jm(**enc).logits[:, -1, :].float(), -1); a, b = pr[:, AID], pr[:, BID]
            sb[idx] = (b / (a + b)).cpu().numpy(); mass[idx] = (a + b).cpu().numpy()
        out = df.drop(columns=[c for c in ("prompt", "text") if c in df.columns]).copy(); out["score_B"], out["format_mass"] = sb, mass
        out.to_csv(f, index=False); cm.log(f"judged {name}: {len(out)}")
    if (R3 / "m1_generations.csv").exists(): judge(pd.read_csv(R3 / "m1_generations.csv"), lambda r: r.axis, "m1")
    if (R3 / "m4_validation_texts.csv").exists(): judge(pd.read_csv(R3 / "m4_validation_texts.csv"), lambda r: r.dim, "m4_validation")
    cf = pd.read_csv(OUT / "counterfactual_generations.csv"); cf = cf[cf.design == "year_swap"]
    of = pd.read_csv(OUT / "offset_generations.csv"); of = of[of.cond.isin(["C4_nonpolitical", "C7_nonpolitical_1955"])]
    for dim in ("tone", "formality"):
        judge(cf, lambda r, d=dim: d, f"placebo_{dim}_yearswap"); judge(of, lambda r, d=dim: d, f"placebo_{dim}_ordinary")
(OUT / "step25.done").touch(); cm.log("step 25 done")
