"""Study I shared helpers (PLAN v0.1). Reuses Study H (h_common: new persona bases, no-persona arm, project_run), Study G (g_common:
directions, placebo, score_final) and Study F (f_common) helpers. Study G calibration (layer 14, e' with c_e, original orth_s,
HALF k per axis) is used without recalibration."""
import json, os, sys
from pathlib import Path
import numpy as np, torch

CODE = Path(__file__).resolve().parent
I = CODE.parent
sys.path.insert(0, str(I.parent / "STUDY_H_dose" / "code"))
import h_common as hc                        # noqa: E402
gc, fc = hc.gc, hc.fc
OUT = Path(os.environ.get("STUDYI_OUT", I / "out")); KEY = "qwen"
GEN_CONDS = ["NONE", "TA_H", "AT_H"]
CELLS = {"TT": (1, 1), "TA": (1, -1), "AT": (-1, 1), "AA": (-1, -1)}
log, seed_for, atomic_csv, atomic_json, chat, items, lens_run, decision_layer, hbits = (
    hc.log, hc.seed_for, hc.atomic_csv, hc.atomic_json, hc.chat, hc.items, hc.lens_run, hc.decision_layer, hc.hbits)
Steer, Capture, load, project_run, NOPERSONA = hc.Steer, hc.Capture, hc.load, hc.project_run, hc.NOPERSONA

def out_dir(*parts):
    p = OUT.joinpath(KEY, *parts); p.mkdir(parents=True, exist_ok=True); return p

def personas(kind="confirm"):
    return hc.personas(kind)

def calibration():
    """Study G chosen configuration: layer, mu, c (e'), HALF k per axis"""
    G = hc.g_calibration(); assert G["vtype"] == "response_avg" and G["HALF"]["econ"]["ok"] and G["HALF"]["soc"]["ok"]
    return dict(layer=int(G["layer"]), mu=float(G["mu"]), c_e=float(G["c"]), ke=float(G["HALF"]["econ"]["k"]), ks=float(G["HALF"]["soc"]["k"]))

def conditions():
    out = [dict(name="NONE", kind="none", strength="none", de=0.0, ds=0.0, plac=-1, scale=0.0)]
    for sname, sc in [("H", 1.0), ("L", 0.5)]:
        for n, (de, ds) in CELLS.items(): out.append(dict(name=f"{n}_{sname}", kind="cell", strength=sname, de=float(de), ds=float(ds), plac=-1, scale=sc))
    out.append(dict(name="PLAC_H_0", kind="plac", strength="H", de=0.0, ds=0.0, plac=0, scale=1.0))
    out.append(dict(name="PLAC_H_1", kind="plac", strength="H", de=0.0, ds=0.0, plac=1, scale=1.0))
    out.append(dict(name="PLAC_L_0", kind="plac", strength="L", de=0.0, ds=0.0, plac=0, scale=0.5))
    return out

def conditions_a5():
    """A2 (2026-10-16, Codex audit): placebos with the norm of the actual cross-cell push |ke e' - ks s| mu (x scale), same random directions"""
    return [dict(name="PLACX_H_0", kind="placx", strength="H", de=0.0, ds=0.0, plac=0, scale=1.0), dict(name="PLACX_H_1", kind="placx", strength="H", de=0.0, ds=0.0, plac=1, scale=1.0),
            dict(name="PLACX_L_0", kind="placx", strength="L", de=0.0, ds=0.0, plac=0, scale=0.5)]

class Pusher:
    """w = mu * own * scale * (de * ke * e' + ds * ks * orth_s); placebo = scale * hypot(ke, ks) * mu * random direction orthogonal to both axes"""
    def __init__(self, C, oe, os_):
        self.mu, self.ke, self.ks = C["mu"], C["ke"], C["ks"]; self.oe, self.os_ = oe, os_
        self.ep = hc.orth(oe, os_, C["c_e"]); self.sp = os_
    def w(self, cond, own):
        if cond["kind"] == "none": return torch.zeros_like(self.oe)
        if cond["kind"] == "plac": return cond["scale"] * float(np.hypot(self.ke, self.ks)) * self.mu * gc.placebo(self.oe, self.os_, cond["plac"])
        if cond["kind"] == "placx": return cond["scale"] * float((self.ke * self.ep - self.ks * self.sp).norm()) * self.mu * gc.placebo(self.oe, self.os_, cond["plac"])   # A2
        return self.mu * own * cond["scale"] * (cond["de"] * self.ke * self.ep + cond["ds"] * self.ks * self.sp)
