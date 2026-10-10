import csv
import re
import sys
from html import escape, unescape
from pathlib import Path

import mistune

root = Path(__file__).resolve().parent
name = sys.argv[1] if len(sys.argv) > 1 else "MASTERPLAN"
source = (root / f"{name}.md").read_text()
render = mistune.create_markdown(escape=True, plugins=["table"])
body = render(source)
toc = []
used = set()


def add_heading_id(match):
    level, title = match.groups()
    plain = unescape(re.sub(r"<[^>]+>", "", title))
    slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")
    while slug in used:
        slug += "-section"
    used.add(slug)
    if level == "2":
        toc.append(f'<li><a href="#{slug}">{escape(plain)}</a></li>')
    return f'<h{level} id="{slug}">{title}</h{level}>'


body = re.sub(r"<h([1-3])>(.*?)</h\1>", add_heading_id, body, flags=re.DOTALL)
document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIRR masterplan — decision quality under shift</title>
<style>
:root{--ink:#1b2830;--muted:#59656c;--line:#d7dfdf;--paper:#fff;--wash:#f1f5f4;--accent:#125c58}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--wash);color:var(--ink);font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}
.status{padding:13px 24px;background:#153d3c;color:white;font-size:13px;letter-spacing:.03em}
.layout{display:grid;grid-template-columns:260px minmax(0,940px);max-width:1240px;margin:0 auto;gap:24px;padding:30px 20px 70px}
nav{position:sticky;top:20px;align-self:start;font-size:13px;padding:15px 12px}nav p{font-weight:700;color:var(--accent);margin:0 0 12px}nav ul{margin:0;padding:0;list-style:none}nav li{margin-bottom:9px}nav a{color:var(--muted);text-decoration:none}nav a:hover,nav a:focus{color:var(--accent);text-decoration:underline}
main{background:var(--paper);padding:40px 46px;border:1px solid var(--line);border-radius:8px;min-width:0}
h1{font-size:34px;line-height:1.2;margin:0 0 22px;letter-spacing:-.035em}h2{font-size:25px;line-height:1.3;margin:45px 0 15px;padding-top:12px;border-top:2px solid var(--line);scroll-margin-top:22px}h3{font-size:19px;margin:27px 0 12px;line-height:1.4}p{margin:14px 0}a{color:var(--accent);overflow-wrap:anywhere}blockquote{margin:23px 0;padding:3px 20px;border-left:4px solid var(--accent);background:#edf5f2;font-size:18px}
table{border-collapse:collapse;display:block;overflow-x:auto;width:100%;margin:22px 0;font-size:13px;line-height:1.55}th,td{text-align:left;vertical-align:top;padding:11px 12px;border:1px solid var(--line);min-width:100px}th{background:#e9f0ee;font-weight:650}td:first-child{min-width:125px}tr:nth-child(even) td{background:#f8faf9}
ul,ol{padding-left:23px}li{margin:7px 0}code{font-size:.9em;background:#eef1f1;padding:2px 4px;border-radius:3px}strong{font-weight:680}.footer{margin-top:40px;border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted)}
@media(max-width:920px){.layout{display:block;padding:14px 12px 40px}nav{position:static;padding:12px 5px 18px}nav ul{columns:2}nav li{break-inside:avoid}main{padding:25px 23px}h1{font-size:29px}}
@media(max-width:520px){nav ul{columns:1}main{padding:22px 17px}h1{font-size:26px}h2{font-size:22px}body{font-size:15px}}
@media print{body{background:white;font-size:11pt}.status,nav{display:none}.layout{display:block;padding:0}main{border:0;padding:0}h2{break-after:avoid}h3{break-after:avoid}table{display:table;font-size:9pt}th,td{min-width:0;padding:6px}a{color:inherit}}
</style></head><body>
<header class="status">RESEARCH-DESIGN DRAFT · 9 OCT 2026 · NO EXPERIMENTS AUTHORISED OR RUN</header>
<div class="layout"><nav aria-label="Contents"><p>AIRR masterplan</p><ul>TOC</ul></nav>
<main>BODY<footer class="footer">Prepared for Gyanateet Dutta. Budget ceilings and timeline are proposals, not measured runs. No cluster access or dataset downloads were performed.</footer></main></div>
</body></html>"""
document = document.replace("TOC", "".join(toc)).replace("BODY", body)
(root / f"{name}.html").write_text(document)

with (root / "budget-ceilings.csv").open() as handle:
    rows = list(csv.DictReader(handle))
assert sum(int(row["cap_NHR"]) for row in rows) == 20_000
assert sum(int(row["normal_rate_GPU_hour_equivalent"]) for row in rows) == 80_000
assert all(
    int(row["normal_rate_GPU_hour_equivalent"]) == 4 * int(row["cap_NHR"])
    for row in rows
)
assert not re.search(r"<script|https?://[^\" ]+\.(?:js|css)", document)
assert "TOC" not in document and "BODY" not in document
print(f"Rendered {len(toc)} sections; budget total 20,000 NHR; {len(document):,} HTML characters.")
