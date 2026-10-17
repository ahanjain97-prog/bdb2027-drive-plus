"""Build the Kaggle submission notebook from submission/writeup.md: markdown cells with figures embedded as data URIs,
then a runnable code appendix that recomputes Drive+ from the competition's combine_tracking.csv.
Also renders an HTML review copy. Output: submission/drive_plus_submission.ipynb, submission/review.html
"""
import base64, re, json, sys
import nbformat as nbf
ROOT = "/Users/ahanjain/bdb_ssl"; SUB = f"{ROOT}/submission"
md = open(f"{SUB}/writeup.md").read()
def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(open(f"{SUB}/{path}", "rb").read()).decode()
# split into cells at figures so each image sits in its own cell (Kaggle renders large base-64 images more reliably that way)
parts = re.split(r"(!\[[^\]]*\]\([^)]*\))", md)
cells = []
for p in parts:
    m = re.match(r"!\[([^\]]*)\]\(([^)]*)\)", p)
    if m:
        cells.append(nbf.v4.new_markdown_cell(f'<img src="{data_uri(m.group(2))}" alt="{m.group(1)}" style="max-width:100%">\n\n*{m.group(1)}*'))
    elif p.strip():
        cells.append(nbf.v4.new_markdown_cell(p.strip()))
code = open(f"{ROOT}/prereg/cb_drive_speed.py").read().split('"""', 2)[2]          # function bodies only
code2 = open(f"{ROOT}/prereg/drive_plus.py").read().split('"""', 2)[2].replace("from cb_drive_speed import cb_drive_speed\n", "")
appendix = f'''# Appendix code: recompute Drive+ from the competition data (frozen definition, SHA-256 in the writeup)
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
{code.strip()}

{code2.strip()}

combine = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2027/combine_tracking.csv")
players = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2027/players.csv").set_index("nfl_id")
grades = drive_plus(combine).join(players[["display_name", "nfl_position"]]).sort_values("drive_plus", ascending=False)
print(grades[grades.nfl_position.isin(["CB", "SS", "FS", "DB"])].head(20).round(1).to_string())
'''
cells.append(nbf.v4.new_markdown_cell("### Appendix code\nThe cell below recomputes every player's Drive+ from `combine_tracking.csv` with the frozen definition. The coverage reps, outcomes and figures are produced by the analysis repository linked above; the scripts are numbered in the order they are described."))
cells.append(nbf.v4.new_code_cell(appendix))
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, f"{SUB}/drive_plus_submission.ipynb")
# review html
import markdown
html_md = re.sub(r"!\[([^\]]*)\]\(([^)]*)\)", lambda m: f'<figure><img src="{data_uri(m.group(2))}" alt="{m.group(1)}"><figcaption>{m.group(1)}</figcaption></figure>', md)
body = markdown.markdown(html_md, extensions=["tables"])
page = f"""<title>Drive+ submission</title>
<style>
:root{{--bg:#fbfbf8;--ink:#141414;--ink2:#4a4a46;--rule:#e3e2da;--hi:#1565c0;color-scheme:light}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#111;--ink:#f2f2ee;--ink2:#bdbdb6;--rule:#2b2b28;--hi:#5b9cf0;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#111;--ink:#f2f2ee;--ink2:#bdbdb6;--rule:#2b2b28;--hi:#5b9cf0;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font-family:Georgia,"Times New Roman",serif;font-size:17px;line-height:1.6;margin:0;padding-inline:clamp(16px,5vw,48px);padding-block:32px 72px}}
.w{{max-width:780px;margin:0 auto}} h1{{font-size:clamp(30px,5vw,44px);line-height:1.1;letter-spacing:-.01em}} h2{{margin-top:2em;font-size:26px}} h3{{font-size:19px}}
p,li{{max-width:70ch}} strong{{font-weight:700}} figure{{margin:28px 0}} img{{max-width:100%;height:auto;border:1px solid var(--rule);border-radius:4px}}
figcaption{{font-family:system-ui,sans-serif;font-size:13px;color:var(--ink2);margin-top:8px;line-height:1.45}} hr{{border:0;border-top:1px solid var(--rule);margin:40px 0}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:13px;word-break:break-all}} .meta{{font-family:system-ui,sans-serif;font-size:13px;color:var(--ink2);border-bottom:1px solid var(--rule);padding-bottom:10px;margin-bottom:24px}}
</style>
<div class="w"><div class="meta">NFL Big Data Bowl 2027 · Open track · writeup draft for review · {sys.argv[1] if len(sys.argv) > 1 else ''}</div>{body}</div>"""
open(f"{SUB}/review.html", "w").write(page)
import os; print("notebook", os.path.getsize(f"{SUB}/drive_plus_submission.ipynb") // 1024, "KB | review.html", os.path.getsize(f"{SUB}/review.html") // 1024, "KB | cells", len(cells))
