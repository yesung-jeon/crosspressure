#!/usr/bin/env python
"""Step 5 - build the self-contained animated map page: out/temporal_payload.json -> out/congress_map.html
Usage: python 05_build_map.py [out_dir]"""
import json, sys
from pathlib import Path
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
payload = open(OUT / "temporal_payload.json").read()
HTML = r'''<title>Congress in Model Space</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#F4F6F8;--panel:#FFFFFF;--ink:#1C2430;--ink2:#4B5563;--muted:#8A94A3;--grid:#E1E6EC;--axis:#B6BEC9;--dem:#2F6FCF;--rep:#D24B2A;--oth:#7B8794;--accent:#1C2430;--tip:#1C2430;--tipink:#F4F6F8;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#12161C;--panel:#181D25;--ink:#E6EAF0;--ink2:#B4BCC8;--muted:#7F8896;--grid:#242B35;--axis:#3A4350;--dem:#5B8EE6;--rep:#E4694A;--oth:#8E98A6;--accent:#E6EAF0;--tip:#E6EAF0;--tipink:#12161C;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#12161C;--panel:#181D25;--ink:#E6EAF0;--ink2:#B4BCC8;--muted:#7F8896;--grid:#242B35;--axis:#3A4350;--dem:#5B8EE6;--rep:#E4694A;--oth:#8E98A6;--accent:#E6EAF0;--tip:#E6EAF0;--tipink:#12161C;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;font-size:14px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-block:20px 40px;padding-inline:16px}
h1{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:clamp(24px,3.4vw,34px);margin:0 0 6px;text-wrap:balance;letter-spacing:-.01em}
.lede{color:var(--ink2);max-width:68ch;margin:0 0 14px}
.controls{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;margin:0 0 12px}
button,select{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--axis);border-radius:6px;padding:6px 12px;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--dem);outline-offset:2px}
button.primary{background:var(--accent);color:var(--bg);border-color:var(--accent);min-width:84px}
label.chk{display:inline-flex;gap:6px;align-items:center;color:var(--ink2)}
.slider{display:flex;align-items:center;gap:10px;flex:1 1 320px;min-width:240px}
input[type=range]{flex:1;accent-color:var(--dem)}
.yr{font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500;font-variant-numeric:tabular-nums;min-width:11ch;text-align:right}
.stage{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px;align-items:start}
@media (max-width:820px){.stage{grid-template-columns:1fr}}
.plot{position:relative;background:var(--panel);border:1px solid var(--grid);border-radius:8px;overflow:hidden}
canvas{display:block;width:100%;height:auto}
.side{display:grid;gap:12px}
.card{background:var(--panel);border:1px solid var(--grid);border-radius:8px;padding:12px 14px}
.card h3{margin:0 0 6px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
.big{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:22px;font-variant-numeric:tabular-nums;line-height:1.2}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-variant-numeric:tabular-nums}
.kv b{font-weight:500;color:var(--ink2)}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--ink2);font-size:13px}
.sw{display:inline-block;width:10px;height:10px;margin-right:6px;vertical-align:-1px}
.sw.dem{background:var(--dem);border-radius:50%}.sw.rep{background:var(--rep)}.sw.oth{width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:10px solid var(--oth);background:none}
.tip{position:absolute;pointer-events:none;background:var(--tip);color:var(--tipink);padding:8px 10px;border-radius:6px;font-size:12px;line-height:1.35;max-width:260px;display:none;z-index:2}
.tip b{font-weight:600}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-top:16px}
.chart h3{margin:0 0 4px;font-size:13px;font-weight:600}
.chart p{margin:0 0 6px;color:var(--muted);font-size:12px}
svg text{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;fill:var(--ink2)}
details{margin-top:16px}summary{cursor:pointer;color:var(--ink2)}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12px;font-family:"IBM Plex Mono",ui-monospace,monospace}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:right}th{color:var(--muted);font-weight:500}
.tblwrap{overflow-x:auto}
.foot{color:var(--muted);font-size:12px;margin-top:18px;max-width:80ch}
@media (prefers-reduced-motion: reduce){.reduce{transition:none}}
</style>
<div class="wrap">
<h1>Congress in Model Space</h1>
<p class="lede">21,288 legislator-Congress observations (U.S. House and Senate, 80th–118th Congress, 1947–2023) projected onto the economic and social ideology directions inside Qwen2.5-7B-Instruct. Each prompt names only the legislator, chamber, state and year, never the party. The origin is the model's own position with no persona.</p>
<div class="controls">
  <button class="primary" id="play">Play</button>
  <div class="slider"><input type="range" id="frame" min="0" max="38" value="0" step="1" aria-label="Congress"><span class="yr" id="yr"></span></div>
  <select id="speed" aria-label="Speed"><option value="900">Slow</option><option value="500" selected>Normal</option><option value="250">Fast</option></select>
  <label class="chk"><input type="checkbox" id="trails" checked> Trails of long-serving members</label>
  <label class="chk"><input type="checkbox" id="known"> Only members whose party the model recalls</label>
</div>
<div class="stage">
  <div class="plot"><canvas id="c" width="860" height="720"></canvas><div class="tip" id="tip"></div></div>
  <div class="side">
    <div class="card"><h3>Congress</h3><div class="big" id="ctitle"></div><div class="kv" id="cinfo"></div></div>
    <div class="card"><h3>Party separation (Cohen's d)</h3><div class="kv" id="dinfo"></div></div>
    <div class="card"><h3>Correlation with NOMINATE dim1</h3><div class="kv" id="rinfo"></div></div>
    <div class="card"><div class="legend"><span><i class="sw dem"></i>Democrats</span><span><i class="sw rep"></i>Republicans</span><span><i class="sw oth"></i>Other</span></div><p style="margin:8px 0 0;color:var(--muted);font-size:12px">Large outlined markers are party means; the line behind them traces earlier Congresses. Hover a point to see the member.</p></div>
  </div>
</div>
<div class="charts">
  <div class="card chart"><h3>Party means: economic axis</h3><p>model units, + = right</p><svg id="s1" viewBox="0 0 340 170" width="100%" role="img" aria-label="Party means on the economic axis over time"></svg></div>
  <div class="card chart"><h3>Party means: social axis</h3><p>model units, + = traditional</p><svg id="s2" viewBox="0 0 340 170" width="100%" role="img" aria-label="Party means on the social axis over time"></svg></div>
  <div class="card chart"><h3>Party separation d</h3><p>model economic and social axes vs NOMINATE dim1</p><svg id="s3" viewBox="0 0 340 170" width="100%" role="img" aria-label="Party separation over time"></svg></div>
</div>
<details><summary>Per-Congress table</summary><div class="tblwrap"><table id="tbl"></table></div></details>
<p class="foot">Vectors: layer-17 mean-difference of hidden states between responses written under paired instructions (economic left vs right, social progressive vs traditional), Löwdin-orthogonalized. Projection: the last-token hidden state of each persona + question prompt, dotted with the vectors and averaged over 5 economic and 5 social questions. NOMINATE: Voteview HSall_members. "Party recalled" = the model named the correct party when given only name, chamber, state and year.</p>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const ord=n=>{const s=['th','st','nd','rd'],v=n%100;return n+(s[(v-20)%10]||s[v]||s[0]);};
const C=D.congresses, N=C.length; const byC=new Map(); C.forEach((c,i)=>byC.set(c.c,i));
const members=Array.from({length:N},()=>[]); D.members.forEach(m=>{members[byC.get(m[0])].push(m);});
const idxMap=members.map(arr=>{const mp=new Map(); arr.forEach(m=>mp.set(m[8],m)); return mp;});
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const cv=document.getElementById('c'), ctx=cv.getContext('2d'); const W=cv.width,H=cv.height; const P={l:56,r:18,t:16,b:46};
const xl=D.xlim,yl=D.ylim; const sx=x=>P.l+(x-xl[0])/(xl[1]-xl[0])*(W-P.l-P.r), sy=y=>H-P.b-(y-yl[0])/(yl[1]-yl[0])*(H-P.t-P.b);
let f=0, playing=false, t0=null, prevF=0, tween=1, hover=null, timer=null;
const $=id=>document.getElementById(id); const slider=$('frame'); slider.max=N-1;
const col=p=>[css('--dem'),css('--rep'),css('--oth')][p];
function mark(x,y,p,r,fill,stroke){ctx.beginPath(); if(p===0){ctx.arc(x,y,r,0,Math.PI*2);} else if(p===1){ctx.rect(x-r,y-r,2*r,2*r);} else {ctx.moveTo(x,y-r*1.2);ctx.lineTo(x+r*1.1,y+r*0.8);ctx.lineTo(x-r*1.1,y+r*0.8);ctx.closePath();}
 if(fill){ctx.fillStyle=fill;ctx.fill();} if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke();}}
function draw(){
 ctx.clearRect(0,0,W,H); ctx.fillStyle=css('--panel'); ctx.fillRect(0,0,W,H);
 // grid + axes
 ctx.strokeStyle=css('--grid'); ctx.lineWidth=1; ctx.font='11px "IBM Plex Mono",monospace'; ctx.fillStyle=css('--ink2'); ctx.textAlign='center';
 const xt=ticks(xl), yt=ticks(yl);
 xt.forEach(v=>{const x=sx(v); ctx.beginPath();ctx.moveTo(x,P.t);ctx.lineTo(x,H-P.b);ctx.stroke(); ctx.fillText(v.toFixed(1),x,H-P.b+16);});
 ctx.textAlign='right'; yt.forEach(v=>{const y=sy(v); ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(W-P.r,y);ctx.stroke(); ctx.fillText(v.toFixed(1),P.l-8,y+4);});
 ctx.strokeStyle=css('--axis'); ctx.lineWidth=1.5; ctx.beginPath(); ctx.moveTo(sx(0),P.t); ctx.lineTo(sx(0),H-P.b); ctx.moveTo(P.l,sy(0)); ctx.lineTo(W-P.r,sy(0)); ctx.stroke();
 ctx.fillStyle=css('--ink2'); ctx.font='500 12px "IBM Plex Sans",sans-serif'; ctx.textAlign='center';
 ctx.fillText('Economic: left ←            → right', (P.l+W-P.r)/2, H-8);
 ctx.save(); ctx.translate(14,(P.t+H-P.b)/2); ctx.rotate(-Math.PI/2); ctx.fillText('Social: progressive ←            → traditional',0,0); ctx.restore();
 ctx.font='11px "IBM Plex Sans",sans-serif'; ctx.textAlign='left'; ctx.fillStyle=css('--muted'); ctx.fillText('model default (0,0)', sx(0)+6, sy(0)-6);
 // party-mean trails
 const onlyKnown=$('known').checked;
 [['dem',0],['rep',1]].forEach(([k,p])=>{ctx.strokeStyle=col(p); ctx.lineWidth=2; ctx.globalAlpha=.45; ctx.beginPath(); for(let i=0;i<=f;i++){const q=C[i][k]; if(i===0)ctx.moveTo(sx(q[0]),sy(q[1])); else ctx.lineTo(sx(q[0]),sy(q[1]));} ctx.stroke(); ctx.globalAlpha=1;});
 // named trails
 if($('trails').checked){ctx.setLineDash([3,3]); D.trails.forEach(tr=>{const pts=tr.pts.filter(q=>byC.get(q[0])<=f); if(pts.length<2)return; ctx.strokeStyle=col(tr.party); ctx.lineWidth=1.2; ctx.globalAlpha=.7; ctx.beginPath(); pts.forEach((q,i)=>{i?ctx.lineTo(sx(q[1]),sy(q[2])):ctx.moveTo(sx(q[1]),sy(q[2]));}); ctx.stroke(); ctx.globalAlpha=1;
   const last=pts[pts.length-1]; if(byC.get(last[0])===f){ctx.fillStyle=css('--ink'); ctx.font='11px "IBM Plex Sans",sans-serif'; ctx.textAlign='left'; ctx.fillText(tr.name, sx(last[1])+7, sy(last[2])+4);} }); ctx.setLineDash([]);}
 // members (tweened from previous frame)
 const cur=members[f], prev=idxMap[prevF];
 cur.forEach(m=>{ if(onlyKnown&&!m[9])return; let x=m[2],y=m[3],a=1; const pm=prev.get(m[8]); if(tween<1){ if(pm&&prevF!==f){x=pm[2]+(m[2]-pm[2])*tween; y=pm[3]+(m[3]-pm[3])*tween;} else a=tween; }
   ctx.globalAlpha=.55*a; mark(sx(x),sy(y),m[1],3.2,col(m[1]),null); });
 ctx.globalAlpha=1;
 // party means
 [['dem',0],['rep',1]].forEach(([k,p])=>{const q=C[f][k]; mark(sx(q[0]),sy(q[1]),p,8,css('--panel'),col(p)); mark(sx(q[0]),sy(q[1]),p,4,col(p),null);});
 if(hover){mark(sx(hover[2]),sy(hover[3]),hover[1],5.5,col(hover[1]),css('--panel'));}
}
function ticks(l){const s=(l[1]-l[0])>3?1:0.5; const out=[]; for(let v=Math.ceil(l[0]/s)*s; v<=l[1]+1e-9; v+=s) out.push(+v.toFixed(2)); return out;}
function info(){const c=C[f]; $('yr').textContent=`${c.year}–${c.year+1} · ${ord(c.c)}`; $('ctitle').textContent=`${ord(c.c)} Congress (${c.year})`;
 $('cinfo').innerHTML=`<b>Members</b><span>${c.n}</span><b>Party recalled</b><span>${Math.round(c.known*100)}%</span><b>Dem mean (x, y)</b><span>${c.dem[0].toFixed(2)}, ${c.dem[1].toFixed(2)}</span><b>Rep mean (x, y)</b><span>${c.rep[0].toFixed(2)}, ${c.rep[1].toFixed(2)}</span>`;
 $('dinfo').innerHTML=`<b>Model economic</b><span>${c.d_x.toFixed(2)}</span><b>Model social</b><span>${c.d_y.toFixed(2)}</span><b>NOMINATE dim1</b><span>${c.d_dim1.toFixed(2)}</span>`;
 $('rinfo').innerHTML=`<b>Economic score</b><span>r = ${c.r_x.toFixed(2)}</span><b>Social score</b><span>r = ${c.r_y.toFixed(2)}</span>`; slider.value=f;}
function goto(nf){prevF=f; f=nf; tween=0; hover=null; const start=performance.now(); const dur=matchMedia('(prefers-reduced-motion: reduce)').matches?0:380;
 function step(now){tween=dur?Math.min(1,(now-start)/dur):1; draw(); if(tween<1) requestAnimationFrame(step);} requestAnimationFrame(step); info(); drawCharts();}
slider.addEventListener('input',()=>goto(+slider.value));
$('play').addEventListener('click',()=>{playing=!playing; $('play').textContent=playing?'Pause':'Play'; if(playing){ if(f>=N-1) goto(0); tick(); } else clearTimeout(timer);});
function tick(){ if(!playing) return; timer=setTimeout(()=>{ if(f<N-1){goto(f+1); tick();} else {playing=false; $('play').textContent='Play';} }, +$('speed').value); }
$('trails').addEventListener('change',draw); $('known').addEventListener('change',()=>{draw();});
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(); const mx=(e.clientX-r.left)*W/r.width, my=(e.clientY-r.top)*H/r.height; let best=null,bd=100; const onlyKnown=$('known').checked;
 members[f].forEach(m=>{ if(onlyKnown&&!m[9])return; const dx=sx(m[2])-mx, dy=sy(m[3])-my, d=dx*dx+dy*dy; if(d<bd){bd=d;best=m;} });
 hover=best; draw(); const tip=$('tip'); if(best){const c=C[f]; tip.style.display='block'; tip.style.left=Math.min(e.clientX-r.left+14, r.width-270)+'px'; tip.style.top=(e.clientY-r.top+14)+'px';
  tip.innerHTML=`<b>${best[5]}</b> (${['Dem','Rep','Other'][best[1]]}, ${best[6]}, ${best[7]==='H'?'House':'Senate'}, ${c.year})<br>economic ${best[2].toFixed(2)} · social ${best[3].toFixed(2)}<br>NOMINATE dim1 ${best[4].toFixed(2)} · party recalled: ${best[9]?'yes':'no'}`;} else tip.style.display='none';});
cv.addEventListener('mouseleave',()=>{hover=null; $('tip').style.display='none'; draw();});
// small charts
function line(svg,series,ylabel,zero){const w=340,h=170,l=34,r=8,t=10,b=22; const xs=C.map(c=>c.year); const all=series.flatMap(s=>s.v.filter(Number.isFinite)); let lo=Math.min(...all),hi=Math.max(...all); if(zero){lo=Math.min(lo,0);hi=Math.max(hi,0);} const pad=(hi-lo)*.08||1; lo-=pad;hi+=pad;
 const X=v=>l+(v-xs[0])/(xs[xs.length-1]-xs[0])*(w-l-r), Y=v=>h-b-(v-lo)/(hi-lo)*(h-t-b); let g='';
 [lo+pad,(lo+hi)/2,hi-pad].forEach(v=>{g+=`<line x1="${l}" x2="${w-r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${l-4}" y="${Y(v)+3}" text-anchor="end">${v.toFixed(1)}</text>`;});
 if(zero) g+=`<line x1="${l}" x2="${w-r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)"/>`;
 [1950,1970,1990,2010].forEach(y=>{g+=`<text x="${X(y)}" y="${h-6}" text-anchor="middle">${y}</text>`;});
 series.forEach(s=>{const d=s.v.map((v,i)=>Number.isFinite(v)?`${i?'L':'M'}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`:'').join(' '); g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="2" stroke-dasharray="${s.dash||''}" />`; const li=s.v.length-1; g+=`<text x="${X(xs[li])-2}" y="${Y(s.v[li])-5}" text-anchor="end" fill="var(--ink)">${s.n}</text>`;});
 const cx=X(C[f].year); g+=`<line x1="${cx}" x2="${cx}" y1="${t}" y2="${h-b}" stroke="var(--ink)" stroke-opacity=".5" stroke-dasharray="2,3"/>`;
 svg.innerHTML=g;}
function drawCharts(){const dem=css('--dem'),rep=css('--rep'),oth=css('--oth');
 line($('s1'),[{n:'Dem',v:C.map(c=>c.dem[0]),c:dem},{n:'Rep',v:C.map(c=>c.rep[0]),c:rep}],'',true);
 line($('s2'),[{n:'Dem',v:C.map(c=>c.dem[1]),c:dem},{n:'Rep',v:C.map(c=>c.rep[1]),c:rep}],'',true);
 line($('s3'),[{n:'Economic',v:C.map(c=>c.d_x),c:css('--ink')},{n:'Social',v:C.map(c=>c.d_y),c:css('--ink'),dash:'4,3'},{n:'NOMINATE',v:C.map(c=>c.d_dim1),c:oth}],'',true);}
function table(){const th='<tr><th>Congress</th><th>Year</th><th>n</th><th>Recall</th><th>Dem econ</th><th>Rep econ</th><th>Dem social</th><th>Rep social</th><th>d econ</th><th>d social</th><th>d dim1</th><th>r econ</th><th>r social</th></tr>';
 $('tbl').innerHTML=th+C.map(c=>`<tr><td>${c.c}</td><td>${c.year}</td><td>${c.n}</td><td>${Math.round(c.known*100)}%</td><td>${c.dem[0].toFixed(2)}</td><td>${c.rep[0].toFixed(2)}</td><td>${c.dem[1].toFixed(2)}</td><td>${c.rep[1].toFixed(2)}</td><td>${c.d_x.toFixed(2)}</td><td>${c.d_y.toFixed(2)}</td><td>${c.d_dim1.toFixed(2)}</td><td>${c.r_x.toFixed(2)}</td><td>${c.r_y.toFixed(2)}</td></tr>`).join('');}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{draw();drawCharts();});
f=N-1; info(); draw(); drawCharts(); table();
</script>
'''
open(OUT / "congress_map.html", "w").write(HTML.replace("__DATA__", payload))
print("wrote", OUT / "congress_map.html", (OUT / "congress_map.html").stat().st_size // 1024, "KB")
