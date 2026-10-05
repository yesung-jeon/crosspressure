#!/usr/bin/env python
"""Step 22 - one self-contained HTML (Kim K20): Kim's congress map as the model sees it vs corrected for the model's lens.

Toggle: raw (origin = the no-persona model; Kim's definition) / corrected (origin = ordinary people of the same year) /
time-removed. Slider over Congresses 80-118, play button, Southern Democrats highlighted, party centroids with trails.
Output: out/corrected_map.html
Usage: python 22_build_corrected_map.py [--out out]
"""
import argparse, glob, json
import numpy as np, pandas as pd
import common as cm

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); args = ap.parse_args()
OUT = cm.HERE / args.out
M = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "corrected" / "members_*.csv")))], ignore_index=True)
M = M[M.party.isin(["Democrat", "Republican"])]
M["sd"] = ((M.party == "Democrat") & M.state.isin(cm.SOUTH)).astype(int)
maps = {"raw": ("x_raw", "y_raw"), "corrected": ("x_c", "y_c"), "perp": ("x_perp", "y_perp")}
data = {"congresses": sorted(int(c) for c in M.congress.unique()), "maps": {}}
for k, (xc, yc) in maps.items():
    pts = {int(c): [[round(float(a), 3), round(float(b), 3), 1 if p == "Republican" else 0, int(s), n] for a, b, p, s, n in zip(g[xc], g[yc], g.party, g.sd, g.name)]
           for c, g in M.groupby("congress")}
    cen = {int(c): {p[0]: [round(float(g[g.party == p][xc].mean()), 3), round(float(g[g.party == p][yc].mean()), 3)] for p in ("Democrat", "Republican")} for c, g in M.groupby("congress")}
    lo = np.percentile(np.r_[M[xc], M[yc]], 0.5); data["maps"][k] = dict(points=pts, centroids=cen,
        xr=[float(np.percentile(M[xc], .5)), float(np.percentile(M[xc], 99.5))], yr=[float(np.percentile(M[yc], .5)), float(np.percentile(M[yc], 99.5))])
HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Corrected Congress Map</title><style>
:root{--bg:#fbfaf7;--fg:#1d1d1f;--mut:#6b6b70;--grid:#e3e1db;--card:#fff;--acc:#7a3ff2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141416;--fg:#ececee;--mut:#9a9aa2;--grid:#2c2c31;--card:#1c1c20;--acc:#b391ff}}
:root[data-theme="dark"]{--bg:#141416;--fg:#ececee;--mut:#9a9aa2;--grid:#2c2c31;--card:#1c1c20;--acc:#b391ff}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0}p{color:var(--mut);margin:4px 0 10px}.ctl{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:5px 12px;cursor:pointer}button.off{background:var(--grid);color:var(--fg)}
input[type=range]{width:min(420px,70vw)}.yr{font-size:26px;font-weight:600}.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:8px}
svg{width:100%;height:auto;display:block}#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--grid);border-radius:6px;padding:5px 8px;font-size:12px;display:none}
</style></head><body><main><h1>Kim's congress map: as the model sees it, and corrected</h1>
<p>Each dot is a member of Congress role-played by Qwen2.5-7B-Instruct (Kim's persona: name, chamber, state, year; no party), placed by its economic and social projections. <b>As the model sees it</b>: origin = the assistant itself. <b>Corrected</b>: origin = ordinary named people of the same year, so a dot shows how far right / traditional of an ordinary American of the time the model places the member. <b>Time removed</b>: the axis component along a time direction learned from non-political people is removed.</p>
<div class="ctl"><button id="b_raw">As the model sees it</button><button id="b_corrected">Corrected (same-year ordinary people)</button><button id="b_perp">Time removed</button></div>
<div class="ctl"><button id="play">Play</button><input id="sl" type="range" min="0" step="1"><span class="yr" id="yr"></span><label><input type="checkbox" id="sd" checked> highlight Southern Democrats</label></div>
<div class="card"><svg id="sv" viewBox="0 0 640 520"></svg></div><p id="nt"></p></main><div id="tip"></div>
<script>
const D=__DATA__;let mode="corrected";const $=s=>document.querySelector(s),sl=$("#sl"),tip=$("#tip");sl.max=D.congresses.length-1;sl.value=D.congresses.length-1;
const yr=c=>1789+2*(c-1);
function draw(){const M=D.maps[mode],c=D.congresses[+sl.value],W=640,H=520,m=46;
 const X=v=>m+(v-M.xr[0])/(M.xr[1]-M.xr[0])*(W-2*m),Y=v=>H-m-(v-M.yr[0])/(M.yr[1]-M.yr[0])*(H-2*m);
 let s=`<rect x="${m}" y="${m}" width="${W-2*m}" height="${H-2*m}" fill="none" stroke="var(--grid)"/>`;
 if(0>=M.xr[0]&&0<=M.xr[1])s+=`<line x1="${X(0)}" y1="${m}" x2="${X(0)}" y2="${H-m}" stroke="var(--grid)" stroke-dasharray="4 4"/>`;
 if(0>=M.yr[0]&&0<=M.yr[1])s+=`<line x1="${m}" y1="${Y(0)}" x2="${W-m}" y2="${Y(0)}" stroke="var(--grid)" stroke-dasharray="4 4"/>`;
 s+=`<text x="${W-m}" y="${H-12}" text-anchor="end" fill="var(--mut)" font-size="11">economic right &rarr;</text><text x="10" y="${m-12}" fill="var(--mut)" font-size="11">&uarr; socially traditional</text>`;
 const hl=$("#sd").checked;
 for(const [x,y,r,sd,n] of M.points[c]){const col=r?"#d1495b":(hl&&sd?"#f2a541":"#3a7bd5");s+=`<circle cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="3" fill="${col}" fill-opacity=".55" data-t="${n} (${r?"R":(sd?"D, South":"D")}, ${yr(c)})"/>`}
 for(const p of ["D","R"]){let path="";for(const cc of D.congresses){if(cc>c)break;const [a,b]=M.centroids[cc][p];path+=(path?"L":"M")+X(a).toFixed(1)+","+Y(b).toFixed(1)}
  const [a,b]=M.centroids[c][p];s+=`<path d="${path}" fill="none" stroke="${p=="R"?"#8b1e2d":"#1c4f9c"}" stroke-width="2"/><circle cx="${X(a)}" cy="${Y(b)}" r="7" fill="${p=="R"?"#8b1e2d":"#1c4f9c"}" stroke="#fff" stroke-width="2"/>`}
 $("#sv").innerHTML=s;$("#yr").textContent=yr(c);for(const k of ["raw","corrected","perp"])$("#b_"+k).className=k==mode?"":"off";
 $("#nt").textContent=`${M.points[c].length} members. Large dots and lines: party centroids since 1947. Orange: Southern Democrats. Exploratory: int8 model; corrections described in 20_corrected_map.py.`}
for(const k of ["raw","corrected","perp"])$("#b_"+k).onclick=()=>{mode=k;draw()};sl.oninput=draw;$("#sd").onchange=draw;
let tm=null;$("#play").onclick=()=>{if(tm){clearInterval(tm);tm=null;$("#play").textContent="Play";return}$("#play").textContent="Pause";tm=setInterval(()=>{sl.value=(+sl.value+1)%D.congresses.length;draw()},700)};
document.addEventListener("mousemove",e=>{const t=e.target.closest("[data-t]");if(t){tip.innerHTML=t.dataset.t;tip.style.display="block";tip.style.left=Math.min(e.clientX+12,innerWidth-200)+"px";tip.style.top=(e.clientY+12)+"px"}else tip.style.display="none"});draw();
</script></body></html>"""
(OUT / "corrected_map.html").write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
cm.log(f"saved {OUT / 'corrected_map.html'} ({len(M)} member-Congresses)")
