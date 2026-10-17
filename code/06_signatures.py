"""Build player x part signatures for a per-play block (phys | ssl | surprise)."""
import sys, time
import numpy as np
from sig_utils import load_meta, signatures

block = sys.argv[1]
src = {"phys": "data/blocks/phys_play.npy", "ssl": "data/blocks/ssl_emb.npy", "surprise": "data/blocks/ssl_surprise.npy",
       "sslasym": "data/blocks/sslasym_emb.npy"}[block]
t0 = time.time()
meta = load_meta()
E = np.load(f"/Users/ahanjain/bdb_ssl/{src}").astype(np.float32)
sig = signatures(E, meta, block)
sig.to_parquet(f"/Users/ahanjain/bdb_ssl/data/sig/{block}.parquet", index=False)
print(block, sig.shape, sig.groupby(["part", "kind"]).size().to_dict(), f"{time.time()-t0:.0f}s")
