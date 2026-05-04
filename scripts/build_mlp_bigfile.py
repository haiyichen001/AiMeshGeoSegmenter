"""合并 MLP 20K NPZ → 单个 memmap 大文件, 训练时 1 秒加载"""
import os, numpy as np
from pathlib import Path

ROOT = Path(r"D:\AiMeshGeoSegmenter")
MLP_DIR = ROOT / "data" / "mlp_edges"
BIG = ROOT / "data" / "mlp_big"

files = sorted(f for f in os.listdir(MLP_DIR) if f.endswith('.npz'))
print(f"Reading {len(files)} files...")

pos, neg = [], []
for i, fn in enumerate(files):
    d = np.load(MLP_DIR / fn)
    X, y = d['X'], d['y']
    mp, mn = y == 1, y == 0
    n = min(mp.sum(), mn.sum())
    if n == 0: continue
    pi = np.random.choice(np.where(mp)[0], min(n, 3000), replace=False)
    neg_dih = X[mn, 0]; ni_all = np.where(mn)[0]
    ni = ni_all[np.argsort(X[ni_all, 0])[:min(n, 500)]]
    pos.append(X[pi]); neg.append(X[ni])
    if (i+1) % 4000 == 0: print(f"  {i+1}/{len(files)}")

Xp = np.vstack(pos); Xn = np.vstack(neg)
n = min(len(Xp), len(Xn))
X_all = np.vstack([Xp[:n], Xn[:n]]).astype(np.float32)
y_all = np.array([1]*n + [0]*n, dtype=np.int8)
idx = np.random.permutation(len(X_all))
X_all, y_all = X_all[idx], y_all[idx]

os.makedirs(BIG, exist_ok=True)
np.save(BIG / "X.npy", X_all)
np.save(BIG / "y.npy", y_all)
print(f"Saved: {len(X_all)} samples, {X_all.shape[1]} dims, {X_all.nbytes/1e6:.0f} MB")
