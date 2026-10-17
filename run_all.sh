#!/bin/zsh
# Full pipeline. Raw data: ~/Downloads/nfl-big-data-bowl-2027/. ~2 h on an M5 (pretraining dominates).
set -e
cd "$(dirname "$0")/code"
PY=../.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4   # keep GPU memory <= half of RAM
$PY 00a_to_parquet.py                 # CSV -> typed parquet
$PY 00_inspect.py > ../results/00_inspect.txt
$PY 01_build_sequences.py             # 311,716 player-plays x 80 frames, play-standardized
$PY 02_pretrain.py --epochs 6 --bs 256 --lr 4e-4 --name ssl_v1   # masked-trajectory transformer
$PY 03_embed.py ssl_v1                # per-play embeddings + surprise
$PY 04_physics.py                     # hand-built physics baseline block
$PY 05_targets.py                     # outcome targets + draft-time baseline
for b in phys ssl surprise; do $PY 06_signatures.py $b; done   # context-adjusted player signatures
$PY 07_combine_minirocket.py          # combine-drill MiniRocket block
$PY 08b_probe_selftest.py             # harness must pass positive/negative controls
$PY 09_reliability.py                 # stability / fingerprint / novelty (no outcomes)
$PY 08_probe.py --n_perm 200          # outcome probes with permutation nulls
$PY 10_summary.py
$PY 11_interpret.py
