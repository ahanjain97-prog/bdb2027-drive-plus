"""Check every file pinned by a SHA-256 in the pre-registration documents against the file on disk.
Usage: python tools/verify_prereg.py      (exit code 1 if any pinned file is missing or changed)
"""
import hashlib, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
pairs = {}
for md in sorted((ROOT / "prereg").glob("*.md")):
    for line in md.read_text().splitlines():
        for name, h in re.findall(r"`([\w./-]+\.(?:py|csv|md|parquet|json))`[^`]*?`?([0-9a-f]{64})`?", line):
            pairs.setdefault((name, h), md.name)
bad = 0
for (name, h), src in sorted(pairs.items()):
    cands = [ROOT / name, ROOT / "prereg" / name, ROOT / "code" / name]
    f = next((c for c in cands if c.exists()), None)
    if f is None:
        print(f"MISSING  {name:48s} (pinned in {src})"); bad += 1; continue
    ok = hashlib.sha256(f.read_bytes()).hexdigest() == h
    bad += not ok
    print(f"{'ok      ' if ok else 'CHANGED '} {str(f.relative_to(ROOT)):48s} {h[:16]}…  ({src})")
print(f"\n{len(pairs)} pinned files, {bad} problems")
sys.exit(1 if bad else 0)
