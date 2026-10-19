"""Study J shared helpers (PLAN v0.1). Reuses Study I (i_common), H, G and F helpers. Study G configuration (layer 14, e' with c_e,
original orth_s, HALF k per axis) is used without recalibration. New here: ten social topics (items_j.json, sha pinned), single-issue
items, single-axis pushes, and placebos whose norm equals the actual cross-cell norm ||k_e e' - k_s orth_s|| (PLAN section 2)."""
import hashlib, itertools, json, os, sys
from pathlib import Path
import numpy as np, torch

CODE = Path(__file__).resolve().parent
J = CODE.parent
sys.path.insert(0, str(J.parent / "STUDY_I_conflict_side" / "code"))
import i_common as ic                        # noqa: E402
hc, gc, fc = ic.hc, ic.gc, ic.fc
OUT = Path(os.environ.get("STUDYJ_OUT", J / "out")); KEY = "qwen"
ITEMS = J / "items_j.json"; ITEMS_SHA = open(J / "items_j.sha256", encoding="utf-8").read().split()[0]
log, seed_for, atomic_csv, atomic_json, chat, lens_run, decision_layer, hbits = (
    ic.log, ic.seed_for, ic.atomic_csv, ic.atomic_json, ic.chat, ic.lens_run, ic.decision_layer, ic.hbits)
Steer, load, project_run, NOPERSONA, personas, calibration = ic.Steer, ic.load, ic.project_run, ic.NOPERSONA, ic.personas, ic.calibration
CELLS = {"TT": (1, 1), "TA": (1, -1), "AT": (-1, 1), "AA": (-1, -1), "ET": (1, 0), "EA": (-1, 0), "ST": (0, 1), "SA": (0, -1)}
N_PLAC = 3

def out_dir(*parts):
    p = OUT.joinpath(KEY, *parts); p.mkdir(parents=True, exist_ok=True); return p

def items():
    """bundles (60 topics x 2 families) and 16 single-issue items; pkgL = economic-left / progressive side (listed first in order 0)"""
    assert hashlib.sha256(open(ITEMS, "rb").read()).hexdigest() == ITEMS_SHA, "Study J items changed after the plan"
    I = json.load(open(ITEMS, encoding="utf-8")); E, S = I["economic"], I["social"]
    cap = lambda x: x[0].upper() + x[1:]
    bundles = []
    for econ, soc, fam in itertools.product(E, S, ["party_like", "reversed"]):
        sl, sr = ("P", "T") if fam == "party_like" else ("T", "P")
        bundles.append(dict(topic=f"{econ}|{soc}", econ=econ, soc=soc, family=fam, kang=S[soc]["kang_split"], origin=S[soc]["origin"],
                            pkgL=cap(f"{E[econ]['L']} and {S[soc][sl]}"), pkgR=cap(f"{E[econ]['R']} and {S[soc][sr]}")))
    singles = [dict(topic=k, econ=k, soc="", family="single", kang="", origin="", axis="econ", pkgL=cap(v["L"]), pkgR=cap(v["R"])) for k, v in E.items()] + \
              [dict(topic=k, econ="", soc=k, family="single", kang=v["kang_split"], origin=v["origin"], axis="soc", pkgL=cap(v["P"]), pkgR=cap(v["T"])) for k, v in S.items()]
    return bundles, singles

def conditions():
    out = [dict(name="NONE", kind="none", strength="none", de=0.0, ds=0.0, plac=-1)]
    for n, (de, ds) in CELLS.items():
        out.append(dict(name=f"{n}_H", kind="cell" if de and ds else "single", strength="H", de=float(de), ds=float(ds), plac=-1))
    for i in range(N_PLAC): out.append(dict(name=f"PLAC_X_{i}", kind="plac", strength="H", de=0.0, ds=0.0, plac=i))
    return out

def absvec(own, de, ds):
    """push label in absolute terms: L/R economic component, P/T social component, 0 = none"""
    e = "0" if de == 0 else ("L" if de * own > 0 else "R"); s = "0" if ds == 0 else ("P" if ds * own > 0 else "T")
    return e + s

class Pusher:
    """w = mu * own * (de * k_e * e' + ds * k_s * orth_s); placebo = random direction orthogonal to both axes with the cross-cell norm
    ||k_e e' - k_s orth_s|| * mu (Study G/I used hypot(k_e, k_s), which equals this only if e' is orthogonal to orth_s; it is not)"""
    def __init__(self, C, oe, os_):
        self.mu, self.ke, self.ks = C["mu"], C["ke"], C["ks"]; self.oe, self.os_ = oe, os_
        self.ep = hc.orth(oe, os_, C["c_e"]); self.sp = os_
        self.norm_cross = float((self.ke * self.ep - self.ks * self.sp).norm()); self.norm_aligned = float((self.ke * self.ep + self.ks * self.sp).norm())
        self.cos_e_s = float(self.ep @ self.sp)
        for i in range(N_PLAC):
            d = gc.placebo(oe, os_, i); assert abs(float(d.norm()) - 1) < 1e-3 and abs(float(d @ oe)) < 1e-3 and abs(float(d @ os_)) < 1e-3, "placebo direction not unit/orthogonal"
    def info(self):
        return dict(mu=self.mu, ke=self.ke, ks=self.ks, cos_eprime_orths=self.cos_e_s, norm_cross=self.norm_cross, norm_aligned=self.norm_aligned,
                    norm_placebo=self.norm_cross, norm_hypot_studyGI=float(np.hypot(self.ke, self.ks)), norm_single_e=self.ke, norm_single_s=self.ks)
    def w(self, cond, own):
        if cond["kind"] == "none": return torch.zeros_like(self.oe)
        if cond["kind"] == "plac": return self.norm_cross * self.mu * gc.placebo(self.oe, self.os_, cond["plac"])
        return self.mu * own * (cond["de"] * self.ke * self.ep + cond["ds"] * self.ks * self.sp)
