#!/usr/bin/env python
"""Step 5 - "uncertainty map": one self-contained HTML (data embedded, no external libraries; Kim K20).

Left: Kim's coordinates (x = economic, right = right; y = social, up = traditional; origin = the model itself) for every
member with generations, colour = generation entropy, for the selected Congress (window of +-4 Congresses, since the
sample is one member per state every other Congress). Right: tile map of states, colour = mean entropy of that state's
sampled members in the same window. Slider, play button, "known members only" toggle, colour by entropy or party.
Output: out/uncertainty_map.html
Usage: python 05_build_map.py [--out out]
"""
import argparse, json
import numpy as np, pandas as pd
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out
G = pd.read_csv(OUT / "member_generations.csv"); G = G[G.party.isin(["Democrat", "Republican"])]
M = G.groupby(["icpsr", "congress"]).agg(entropy=("entropy64", "mean"), **{c: (c, "first") for c in
        ["name", "party", "chamber", "state", "year", "known", "x", "y"]}).reset_index()
FRAMES = list(range(80, 119, 2)); W = 4
members = [dict(c=int(r.congress), n=r["name"], p=r.party[0], s=r.state, ch=r.chamber[0], k=int(bool(r.known)),
                x=round(float(r.x), 3), y=round(float(r.y), 3), e=round(float(r.entropy), 4)) for _, r in M.iterrows()]
states = {}
for f in FRAMES:
    w = M[(M.congress >= f - W) & (M.congress <= f + W)]
    for kn in (0, 1):
        ww = w[w.known.astype(bool)] if kn else w
        states[f"{f}_{kn}"] = {s: [round(float(g.entropy.mean()), 4), int(len(g))] for s, g in ww.groupby("state")}
TILES = {"AK": (0, 0), "ME": (11, 0), "VT": (10, 1), "NH": (11, 1), "WA": (1, 2), "ID": (2, 2), "MT": (3, 2), "ND": (4, 2), "MN": (5, 2),
         "IL": (6, 2), "WI": (7, 2), "MI": (8, 2), "NY": (9, 2), "RI": (10, 2), "MA": (11, 2), "OR": (1, 3), "NV": (2, 3), "WY": (3, 3),
         "SD": (4, 3), "IA": (5, 3), "IN": (6, 3), "OH": (7, 3), "PA": (8, 3), "NJ": (9, 3), "CT": (10, 3), "CA": (1, 4), "UT": (2, 4),
         "CO": (3, 4), "NE": (4, 4), "MO": (5, 4), "KY": (6, 4), "WV": (7, 4), "VA": (8, 4), "MD": (9, 4), "DE": (10, 4), "AZ": (2, 5),
         "NM": (3, 5), "KS": (4, 5), "AR": (5, 5), "TN": (6, 5), "NC": (7, 5), "SC": (8, 5), "OK": (4, 6), "LA": (5, 6), "MS": (6, 6),
         "AL": (7, 6), "GA": (8, 6), "HI": (0, 7), "TX": (4, 7), "FL": (9, 7)}
assert len(TILES) == 50
lo, hi = np.percentile(M.entropy, [5, 95])
data = dict(members=members, states=states, frames=FRAMES, tiles=TILES, lo=round(float(lo), 3), hi=round(float(hi), 3),
            xr=[float(np.floor(M.x.min() * 2) / 2), float(np.ceil(M.x.max() * 2) / 2)], yr=[float(np.floor(min(0, M.y.min()) * 2) / 2), float(np.ceil(M.y.max() * 2) / 2)],
            n_members=int(len(M)))

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Uncertainty Map</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1f;--mut:#6b6b70;--grid:#e3e1db;--card:#ffffff;--acc:#7a3ff2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141416;--fg:#ececee;--mut:#9a9aa2;--grid:#2c2c31;--card:#1c1c20;--acc:#b391ff}}
:root[data-theme="dark"]{--bg:#141416;--fg:#ececee;--mut:#9a9aa2;--grid:#2c2c31;--card:#1c1c20;--acc:#b391ff}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0 2px}p.sub{color:var(--mut);margin:0 0 12px}
.ctl{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:8px 0 12px}
.ctl label{color:var(--mut)}input[type=range]{width:min(420px,70vw)}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:5px 12px;cursor:pointer}
.panes{display:grid;grid-template-columns:1.25fr 1fr;gap:14px}@media (max-width:820px){.panes{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px}
.card h2{font-size:14px;margin:0 0 6px}svg{width:100%;height:auto;display:block}
.yr{font-size:28px;font-weight:600}#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--grid);border-radius:6px;padding:6px 8px;font-size:12px;display:none}
.note{color:var(--mut);font-size:12px;margin-top:10px}
</style></head><body><main>
<h1>Uncertainty map: whom does the model hesitate to speak for?</h1>
<p class="sub">Qwen2.5-7B-Instruct role-playing U.S. legislators (Kim's congress_map personas: name, chamber, state, year; no party). Position = Kim's econ x social projection (origin = the model itself). Colour = generation entropy on 10 held-out questions (higher = more hesitant).</p>
<div class="ctl"><button id="play">Play</button><input id="sl" type="range" min="0" step="1"><span class="yr" id="yr"></span>
<label><input type="checkbox" id="kn"> known members only</label>
<label>colour <select id="cm"><option value="e">entropy</option><option value="p">party</option></select></label></div>
<div class="panes"><div class="card"><h2>Members in the model's ideology space (window of &plusmn;4 Congresses)</h2><svg id="sc" viewBox="0 0 620 480"></svg></div>
<div class="card"><h2>Mean hesitation of each state's sampled members</h2><svg id="tm" viewBox="0 0 600 420"></svg><div id="leg"></div></div></div>
<p class="note" id="nt"></p></main><div id="tip"></div>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s), sl=$("#sl"), tip=$("#tip");
sl.max=D.frames.length-1; sl.value=D.frames.length-1;
const yr=c=>1789+2*(c-1);
function col(e){let t=Math.max(0,Math.min(1,(e-D.lo)/(D.hi-D.lo)));const a=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
 const i=Math.min(3,Math.floor(t*4)),f=t*4-i,c=a[i].map((v,j)=>Math.round(v+(a[i+1][j]-v)*f));return `rgb(${c})`}
const pcol=p=>p==="R"?"#d1495b":"#3a7bd5";
function sc(){const f=D.frames[+sl.value],kn=$("#kn").checked,mode=$("#cm").value,W=620,H=480,m=44;
 const X=v=>m+(v-D.xr[0])/(D.xr[1]-D.xr[0])*(W-2*m), Y=v=>H-m-(v-D.yr[0])/(D.yr[1]-D.yr[0])*(H-2*m);
 let s=`<rect x="${m}" y="${m}" width="${W-2*m}" height="${H-2*m}" fill="none" stroke="var(--grid)"/>`;
 s+=`<line x1="${X(0)}" y1="${m}" x2="${X(0)}" y2="${H-m}" stroke="var(--grid)" stroke-dasharray="4 4"/><line x1="${m}" y1="${Y(0)}" x2="${W-m}" y2="${Y(0)}" stroke="var(--grid)" stroke-dasharray="4 4"/>`;
 s+=`<text x="${X(0)+4}" y="${Y(0)-4}" fill="var(--mut)" font-size="11">model default</text>`;
 s+=`<text x="${W-m}" y="${H-10}" fill="var(--mut)" font-size="11" text-anchor="end">economic right &rarr;</text><text x="12" y="${m-10}" fill="var(--mut)" font-size="11">&uarr; socially traditional</text>`;
 const pts=D.members.filter(d=>Math.abs(d.c-f)<=4&&(!kn||d.k));
 for(const d of pts){s+=`<circle cx="${X(d.x).toFixed(1)}" cy="${Y(d.y).toFixed(1)}" r="4" fill="${mode==="e"?col(d.e):pcol(d.p)}" fill-opacity=".85" stroke="${mode==="e"?pcol(d.p):"none"}" stroke-width="1" data-t="${d.n} (${d.p}-${d.s}, ${yr(d.c)})<br>entropy ${d.e.toFixed(3)}${d.k?"":" &middot; party not recalled"}"/>`}
 $("#sc").innerHTML=s; $("#yr").textContent=yr(f); $("#nt").textContent=`${pts.length} members in this window. Outline colour = party (blue D, red R). Exploratory: int8 model, one sample per question, first 64 generated tokens.`}
function tm(){const f=D.frames[+sl.value],kn=$("#kn").checked?1:0,st=D.states[f+"_"+kn]||{},S=48;let s="";
 for(const [k,[c,r]] of Object.entries(D.tiles)){const v=st[k];s+=`<g data-t="${k}: ${v?("mean entropy "+v[0].toFixed(3)+" ("+v[1]+" members)"):"no sampled member"}"><rect x="${10+c*S}" y="${10+r*S}" width="${S-4}" height="${S-4}" rx="6" fill="${v?col(v[0]):"var(--grid)"}"/><text x="${10+c*S+(S-4)/2}" y="${10+r*S+(S-4)/2+4}" text-anchor="middle" font-size="12" fill="${v&&(v[0]-D.lo)/(D.hi-D.lo)>.6?"#111":"#fff"}">${k}</text></g>`}
 $("#tm").innerHTML=s;
 $("#leg").innerHTML=`<svg viewBox="0 0 600 34"><defs><linearGradient id="g">${[0,.25,.5,.75,1].map(t=>`<stop offset="${t}" stop-color="${col(D.lo+t*(D.hi-D.lo))}"/>`).join("")}</linearGradient></defs><rect x="10" y="4" width="300" height="12" fill="url(#g)" rx="3"/><text x="10" y="30" font-size="11" fill="var(--mut)">${D.lo} (confident)</text><text x="310" y="30" font-size="11" fill="var(--mut)" text-anchor="end">${D.hi} (hesitant)</text></svg>`}
function draw(){sc();tm()}
sl.oninput=draw;$("#kn").onchange=draw;$("#cm").onchange=draw;
let timer=null;$("#play").onclick=()=>{if(timer){clearInterval(timer);timer=null;$("#play").textContent="Play";return}
 $("#play").textContent="Pause";timer=setInterval(()=>{sl.value=(+sl.value+1)%D.frames.length;draw()},900)};
document.addEventListener("mousemove",ev=>{const t=ev.target.closest("[data-t]");if(t){tip.innerHTML=t.dataset.t;tip.style.display="block";tip.style.left=Math.min(ev.clientX+12,innerWidth-220)+"px";tip.style.top=(ev.clientY+12)+"px"}else tip.style.display="none"});
draw();
</script></body></html>"""
(OUT / "uncertainty_map.html").write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
cm.log(f"saved {OUT / 'uncertainty_map.html'} ({len(members)} member-Congresses)")
