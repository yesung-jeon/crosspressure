#!/usr/bin/env python
"""S4 sources (PLAN.md): for each coding-sheet row, find the English Wikipedia article through Wikidata (Bioguide id, P1157) and
save its plain-text extract with title, URL, revision id and retrieval date to out/s4/wiki/row<NN>.txt. Read-only GET requests of
public pages; nothing from the study is sent except the Bioguide ids and article titles in the queries.
Usage: python s4_fetch.py"""
import json, sys, time, datetime
import pandas as pd, requests
import dcommon as dc

O = dc.OUT / (sys.argv[1] if len(sys.argv) > 1 else "s4"); W = O / "wiki"; W.mkdir(parents=True, exist_ok=True)   # s4b reuses this script
UA = {"User-Agent": "StudyD-research-script/1.0 (academic research; contact via repository owner)"}
sheet = pd.read_csv(O / "coding_sheet.csv", dtype=str)
ids = [b for b in sheet.bioguide_id.fillna("") if b]
q = "SELECT ?bg ?article WHERE { VALUES ?bg { %s } ?item wdt:P1157 ?bg . ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }" % " ".join(f'"{b}"' for b in ids)
r = requests.get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"}, headers=UA, timeout=60); r.raise_for_status()
title = {}
for b in r.json()["results"]["bindings"]:
    title[b["bg"]["value"]] = requests.utils.unquote(b["article"]["value"].rsplit("/wiki/", 1)[1]).replace("_", " ")
dc.log(f"Wikidata: {len(title)}/{len(ids)} Bioguide ids matched to enwiki")
today = datetime.date.today().isoformat(); led = []
for _, s in sheet.iterrows():
    f = W / f"row{int(s.row):02d}.txt"
    t = title.get(s.bioguide_id or "", "")
    if not t:
        led.append(dict(row=s.row, status="no enwiki via Wikidata")); continue
    if f.exists():
        led.append(dict(row=s.row, status="cached", title=t)); continue
    p = None
    for k in range(5):
        try:
            p = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=60,
                             params={"action": "query", "prop": "extracts|revisions", "explaintext": 1, "titles": t, "redirects": 1, "format": "json", "rvprop": "ids"}).json()
            break
        except (requests.RequestException, ValueError):
            time.sleep(5 * (k + 1))
    if p is None:
        led.append(dict(row=s.row, status="fetch error", title=t)); continue
    page = next(iter(p["query"]["pages"].values()))
    txt = page.get("extract", ""); rev = page.get("revisions", [{}])[0].get("revid", "")
    url = "https://en.wikipedia.org/wiki/" + page.get("title", t).replace(" ", "_")
    f.write_text(f"TITLE: {page.get('title', t)}\nURL: {url}\nREVID: {rev}\nRETRIEVED: {today}\nBIOGUIDE: {s.bioguide_id}\n\n{txt}", encoding="utf-8")
    led.append(dict(row=s.row, status="fetched", title=page.get("title", t), revid=rev, chars=len(txt))); time.sleep(1.5)
pd.DataFrame(led).to_csv(O / "wiki_ledger.csv", index=False)
dc.log(pd.DataFrame(led).status.value_counts().to_dict())
