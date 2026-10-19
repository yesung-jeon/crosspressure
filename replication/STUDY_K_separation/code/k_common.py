"""Study K shared helpers (PLAN v0.1): separation design. Absolute push directions in Kim's Löwdin-orthogonalized coordinates at layer 14:
pure orth_e (E), pure orth_s (S), the EL-ST diagonal D = normalize(orth_e - orth_s), the EL-SP diagonal X = normalize(orth_e + orth_s),
each with both signs, plus Study G's e' (EP) for reference; equal norms (H = Study J cross-cell norm, L = H / sqrt(2)); placebos of the same
norms. The same vector is added for every persona (party enters the analysis only). Items: Study J items without school prayer."""
import os, sys
from pathlib import Path
import numpy as np, torch

CODE = Path(__file__).resolve().parent
K = CODE.parent
sys.path.insert(0, str(K.parent / "STUDY_J_social_topics" / "code"))
import j_common as jc                        # noqa: E402
hc, gc, fc = jc.hc, jc.gc, jc.fc
OUT = Path(os.environ.get("STUDYK_OUT", K / "out")); KEY = "qwen"
EXCL_SOC = ["religion"]
DIRS = ["Ep", "Em", "Sp", "Sm", "Dp", "Dm", "Xp", "Xm"]
N_PERSONAS_PER_PARTY = 15
log, seed_for, atomic_csv, atomic_json, chat, lens_run, decision_layer, hbits = (
    jc.log, jc.seed_for, jc.atomic_csv, jc.atomic_json, jc.chat, jc.lens_run, jc.decision_layer, jc.hbits)
Steer, load, project_run, NOPERSONA, calibration = jc.Steer, jc.load, jc.project_run, jc.NOPERSONA, jc.calibration

def out_dir(*parts):
    p = OUT.joinpath(KEY, *parts); p.mkdir(parents=True, exist_ok=True); return p

def items():
    bundles, singles = jc.items()
    return [b for b in bundles if b["soc"] not in EXCL_SOC], [s for s in singles if s["soc"] not in EXCL_SOC]

def personas(kind="confirm"):
    """15 Democrats + 15 Republicans drawn (seeded) from the Study I/J confirmatory bases; 'confirm+model' adds the no-persona arm"""
    P = jc.personas("confirm"); rng = np.random.default_rng(seed_for("K-personas") % 2**32)
    out = [p for party in ["Democrat", "Republican"] for p in rng.choice([q for q in P if q["party"] == party], N_PERSONAS_PER_PARTY, replace=False)]
    return out + [dict(NOPERSONA)] if kind == "confirm+model" else out

def conditions():
    out = [dict(name="NONE", kind="none", strength="none", vec="NONE", plac=-1)]
    for d in DIRS:
        for s in ("H", "L"): out.append(dict(name=f"{d}_{s}", kind="dir", strength=s, vec=d, plac=-1))
    out.append(dict(name="EP_H", kind="dir", strength="H", vec="EP", plac=-1))
    for i in range(3): out.append(dict(name=f"PLAC_H_{i}", kind="plac", strength="H", vec="PLAC", plac=i))
    out.append(dict(name="PLAC_L_0", kind="plac", strength="L", vec="PLAC", plac=0))
    return out

class Pusher:
    """w = norm(strength) * mu * unit direction; norm H = ||k_e e' - k_s orth_s|| (Study J cross-cell norm), L = H / sqrt(2) so that
    Ep_L + Sm_L = Dp_H exactly (vector sum); placebo directions orthogonal to both axes"""
    def __init__(self, C, oe, os_):
        self.mu = C["mu"]; self.oe, self.os_ = oe, os_; self.ep = hc.orth(oe, os_, C["c_e"])
        self.norm = {"H": float((C["ke"] * self.ep - C["ks"] * os_).norm())}; self.norm["L"] = self.norm["H"] / float(np.sqrt(2))
        n = lambda v: v / v.norm()
        self.dirs = {"Ep": oe, "Em": -oe, "Sp": os_, "Sm": -os_, "Dp": n(oe - os_), "Dm": -n(oe - os_), "Xp": n(oe + os_), "Xm": -n(oe + os_), "EP": self.ep}
        for i in range(3):
            d = gc.placebo(oe, os_, i); assert abs(float(d.norm()) - 1) < 1e-3 and abs(float(d @ oe)) < 1e-3 and abs(float(d @ os_)) < 1e-3
        assert abs(float(oe @ os_)) < 1e-4, "orth_e and orth_s must be orthogonal"
    def info(self):
        return dict(mu=self.mu, norm_H=self.norm["H"], norm_L=self.norm["L"], cos_eprime_orthe=float(self.ep @ self.oe), cos_eprime_orths=float(self.ep @ self.os_),
                    cos_eprime_Dp=float(self.ep @ self.dirs["Dp"]), check_sum=float((self.norm["L"] * (self.dirs["Ep"] + self.dirs["Sm"]) - self.norm["H"] * self.dirs["Dp"]).norm()))
    def w(self, cond):
        if cond["kind"] == "none": return torch.zeros_like(self.oe)
        if cond["kind"] == "plac": return self.norm[cond["strength"]] * self.mu * gc.placebo(self.oe, self.os_, cond["plac"])
        return self.norm[cond["strength"]] * self.mu * self.dirs[cond["vec"]]
