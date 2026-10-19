#!/usr/bin/env python
"""Render paper/manuscript.md to two PDFs: pdf/main.pdf (abstract to references, with Figures 1-3 placed above their captions) and
pdf/appendix.pdf (Appendices A and B). Markdown -> HTML (python-markdown, tables) -> headless Chrome or Edge print-to-PDF.
Nothing in the text is changed here. Usage: python paper/build_pdf.py"""
import re, subprocess, sys
from pathlib import Path
import markdown

P = Path(__file__).resolve().parent; OUT = P / "pdf"; OUT.mkdir(exist_ok=True)
BROWSERS = [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"), Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")]
browser = next((b for b in BROWSERS if b.exists()), None)
assert browser, "no Chrome or Edge found"
t = (P / "manuscript.md").read_text(encoding="utf-8"); a = t.index("# Appendix A")
main, app = t[:a], t[a:]
FIG = {"1": "fig1_map.png", "2": "fig2_profiles.png", "3": "fig3_packages.png"}
for n, f in FIG.items():
    cap = f"**Figure {n}.**"; assert main.count(cap) == 1, cap
    main = main.replace(cap, f'<div class="fig"><img src="{(P / "figures" / f).as_uri()}"></div>\n\n{cap}')
CSS = """
@page { size: Letter; margin: 1in; }
body { font-family: 'Times New Roman', Times, serif; font-size: 11.5pt; line-height: 1.55; color: #000; }
h1 { font-size: 14pt; margin: 1.4em 0 0.5em; } h2 { font-size: 12pt; margin: 1.1em 0 0.4em; }
p { margin: 0 0 0.7em; text-align: justify; }
table { border-collapse: collapse; font-size: 9pt; line-height: 1.3; margin: 0.4em 0 1.2em; width: 100%; page-break-inside: auto; }
th, td { border-top: 0.5pt solid #888; border-bottom: 0.5pt solid #888; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { border-top: 1pt solid #000; border-bottom: 1pt solid #000; }
tr { page-break-inside: avoid; }
.fig { text-align: center; margin: 1em 0 0.3em; page-break-inside: avoid; } .fig img { max-width: 100%; }
.tbl > p:first-child { break-after: avoid; page-break-after: avoid; } .tbl.short { break-inside: avoid; page-break-inside: avoid; } .tbl:not(.short) { break-before: page; page-break-before: always; }
.title { font-size: 9pt; color: #555; margin-bottom: 1.5em; }
"""
def render(md, name, label):
    body = markdown.markdown(md, extensions=["tables"])
    # keep each table caption on the same page as its table; short tables are kept whole
    body = re.sub(r'(<p><strong>Table [^<]*</strong>.*?</p>\s*)(<table>.*?</table>)',
                  lambda m: f'<div class="tbl{" short" if m.group(2).count("<tr>") <= 15 else ""}">{m.group(1)}{m.group(2)}</div>', body, flags=re.S)
    html = f'<!doctype html><html><head><meta charset="utf-8"><title>{label}</title><style>{CSS}</style></head><body>' \
           f'<div class="title">{label} (anonymized manuscript; generated from paper/manuscript.md)</div>{body}</body></html>'
    h = OUT / f"{name}.html"; h.write_text(html, encoding="utf-8"); pdf = OUT / f"{name}.pdf"; pdf.unlink(missing_ok=True)
    r = subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", h.as_uri()],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    assert pdf.exists() and pdf.stat().st_size > 10000, (r.stdout, r.stderr)
    print(name, "->", pdf, f"{pdf.stat().st_size // 1024} KB")
render(main, "main", "Main text")
render(app, "appendix", "Online appendix")
