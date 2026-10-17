"""Build the Drive+ story page: inline the exported JSON (NaN-free) into the HTML template."""
import json, math, sys
ROOT = "/Users/ahanjain/bdb_ssl"
D = json.load(open(f"{ROOT}/results/viz/drive_plus_viz_nfl.json"))
D["spec"] = [s for s in D["spec"] if all(isinstance(s[k], (int, float)) and math.isfinite(s[k]) for k in ("b", "se", "p"))]
def clean(o):
    if isinstance(o, float) and not math.isfinite(o): return None
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    return o
js = json.dumps(clean(D), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
film = json.dumps(json.load(open(f"{ROOT}/results/viz/rep_film.json")), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
out = open(f"{ROOT}/code/97_nfl_page.html").read().replace("/*__DATA__*/", js).replace("/*__FILM__*/", film)
open(sys.argv[1], "w").write(out); print(len(out), "bytes; spec rows", len(D["spec"]))
