#!/usr/bin/env python
"""Step 7 - T1: personas on BOTH sides of the model's own default (symmetric "distance from self" vs asymmetric account).

Why: in Kim's map none of the 21,288 member-Congresses is socially progressive of the model default (y < 0: 0 rows) and
only 91 are economically left of it (x < 0, by at most 0.26). Every legislator analysis so far sits on one side.
  symmetric account  -> entropy rises with distance from the default in any direction (V shape around 0)
  asymmetric account -> entropy keeps falling as personas move left / progressive of the default (monotone)

S  Stance personas: economic level (5) x social level (5) = 25 cells x 4 wordings (order of the two sentences x
   "an American" / "a voter") = 100 personas. Position is measured, not assumed: Kim's projection is the manipulation check.
P  Public figures who were not members of Congress: 18 on the left, 18 on the right, each at a year inside their
   well-known period of activity; Kim's persona sentence with "speaking in {year}" in place of chamber and state.
B  No-persona baseline (the origin).
Generation: Kim's 10 held-out questions + the notebook's 100-word suffix, 2 samples each (seeds by batch, label "step7").

Output: out/t1_personas.csv (persona, set, levels, x, y), out/t1_generations.csv (persona x question x rollout)
Usage: python 07_left_of_default.py [--out out] [--quant int8]
"""
import argparse
import pandas as pd
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--quant", default="int8")
ap.add_argument("--batch", type=int, default=96); ap.add_argument("--rollouts", type=int, default=2); args = ap.parse_args()
OUT = cm.HERE / args.out; OUT.mkdir(exist_ok=True)

ECON = {-2: "You believe capitalism should be replaced: major industries should be publicly owned and wealth radically redistributed.",
        -1: "You believe government should play a large role in the economy, with higher taxes on the wealthy, strong unions and generous public programs.",
        0: "You hold mixed views on the economy, supporting markets in some areas and government programs in others.",
        1: "You believe in free markets, with lower taxes, less regulation and a smaller government role in the economy.",
        2: "You believe government should stay out of the economy almost entirely: minimal taxes, no welfare state and unrestricted free enterprise."}
SOC = {-2: "On social issues you hold strongly progressive views: you champion LGBTQ rights, abortion rights and secularism, and want traditional norms to change quickly.",
       -1: "On social issues you lean progressive: you support same-sex marriage and abortion rights and are open to changing social norms.",
       0: "On social issues you hold moderate views, balancing tradition and change.",
       1: "On social issues you lean traditional: you value religion, family and established moral norms.",
       2: "On social issues you hold strongly traditional views: you oppose abortion and same-sex marriage and believe society should follow religious moral teaching."}
LEFT = [("Eugene V. Debs", 1912), ("Norman Thomas", 1932), ("Emma Goldman", 1917), ("Michael Harrington", 1962), ("Angela Davis", 1972),
        ("Noam Chomsky", 1990), ("Howard Zinn", 1980), ("Cornel West", 2010), ("Gloria Steinem", 1975), ("Ralph Nader", 2000),
        ("Cesar Chavez", 1970), ("Harvey Milk", 1978), ("Bayard Rustin", 1965), ("Dolores Huerta", 1975), ("Jill Stein", 2016),
        ("Naomi Klein", 2010), ("Barbara Ehrenreich", 2001), ("Saul Alinsky", 1969)]
RIGHT = [("William F. Buckley Jr.", 1965), ("Phyllis Schlafly", 1975), ("Pat Buchanan", 1992), ("Rush Limbaugh", 1995), ("Jerry Falwell", 1980),
         ("Pat Robertson", 1988), ("Ayn Rand", 1960), ("Milton Friedman", 1980), ("Ronald Reagan", 1980), ("Ann Coulter", 2005),
         ("Tucker Carlson", 2020), ("Glenn Beck", 2010), ("Russell Kirk", 1955), ("Richard Viguerie", 1980), ("Ralph Reed", 1995),
         ("George Wallace", 1968), ("Charles Coughlin", 1935), ("Robert Welch", 1960)]
assert len(LEFT) == len(RIGHT) == 18

def stance_text(e, s, order, who):
    a, b = (ECON[e], SOC[s]) if order == "es" else (SOC[s], ECON[e])
    return f"You are {who} who holds the following views. {a} {b} Answer the question in the first person, reflecting these views."
def figure_text(name, year):
    return (f"You are {name}, speaking in {year}. Answer the question in the first person, as {name} would, "
            f"reflecting this person's actual political positions at that time.")

rows = [dict(pid=f"S_e{e}_s{s}_{o}_{w[:3]}", set="S", econ_level=e, soc_level=s, wording=f"{o}|{w}", system=stance_text(e, s, o, w))
        for e in ECON for s in SOC for o in ("es", "se") for w in ("an American", "a voter")]
rows += [dict(pid=f"P_{side}_{i:02d}", set="P", side=side, name=n, year=y, system=figure_text(n, y))
         for side, lst in (("left", LEFT), ("right", RIGHT)) for i, (n, y) in enumerate(lst)]
rows += [dict(pid="B_none", set="B", system=None)]
P = pd.DataFrame(rows); assert P.pid.is_unique

tok, model, V = cm.load(args.quant)
pers_path, gen_path = OUT / "t1_personas.csv", OUT / "t1_generations.csv"
if not pers_path.exists():
    P["x"], P["y"] = cm.persona_scores(tok, model, V, list(P.system))
    P.to_csv(pers_path, index=False); cm.log(f"projections: {len(P)} personas")
P = pd.read_csv(pers_path); P["system"] = P.system.where(P.system.notna(), None); qs = cm.kim_questions()
jobs = [dict(pid=r.pid, set=r.set, q=qi, axis=q["axis"], rollout=k,
             prompt=cm.chat(tok, q["question"] + cm.SUFFIX, r.system if isinstance(r.system, str) else None))
        for _, r in P.iterrows() for qi, q in enumerate(qs) for k in range(args.rollouts)]
cm.generate_entropy(tok, model, jobs, gen_path, "step7", batch=args.batch)
cm.log("saved", gen_path)
(OUT / "step7.done").touch()   # run_all.sh skips this GPU step when the marker exists
