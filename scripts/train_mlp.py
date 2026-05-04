"""
MLP Edge Classifier: GPU PyTorch, in-memory from 20K NPZ, k-fold CV, JSON log
"""
import os, json, random, time, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import KFold

ROOT = Path(r"D:\AiMeshGeoSegmenter")
MLP_DIR = ROOT / "data" / "mlp_edges"
MODEL_DIR = ROOT / "models"
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K, EPOCHS, BS = 5, 40, 65536


def auto_tune_batch(model, X, y):
    """Find largest batch size by probing VRAM until >80% or OOM (min 16 steps/epoch)."""
    N = len(X)
    candidates = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    best = 4096
    for bs in candidates:
        if bs > N // 4: break  # too few steps
        try:
            torch.cuda.empty_cache()
            xb = torch.tensor(X[:bs], device=DEVICE)
            yb = torch.tensor(y[:bs], device=DEVICE)
            model.train()
            out = model(xb); l = F.binary_cross_entropy_with_logits(out, yb)
            l.backward()
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            pct = used / total * 100
            best = bs
            del xb, yb, out, l
            if pct > 80:
                print(f"  BS={bs}: {pct:.0f}% VRAM, sufficient")
                break
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                break
            raise
    # Ensure at least 16 steps per epoch
    best = min(best, N // 16)
    return max(best, 4096)
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB, TF32 enabled")

t0 = time.time()
files = sorted(f for f in os.listdir(MLP_DIR) if f.endswith('.npz'))

# Load all into memory
pos, neg = [], []
for i, fn in enumerate(files):
    d = np.load(MLP_DIR / fn); X, y = d['X'], d['y']
    mp, mn = y==1, y==0; n = min(mp.sum(), mn.sum())
    if n == 0: continue
    pi = np.random.choice(np.where(mp)[0], min(n, 200), replace=False)
    ni = np.where(mn)[0]; ni = ni[np.argsort(X[ni,0])[:min(n, 200)]]
    pos.append(X[pi]); neg.append(X[ni])
    if (i+1) % 5000 == 0: print(f"  {i+1}/{len(files)}")

X_all = np.vstack(pos+neg).astype(np.float32)
y_all = np.array([1]*sum(len(p) for p in pos) + [0]*sum(len(n) for n in neg), dtype=np.float32)
idx = np.random.permutation(len(X_all)); X_all, y_all = X_all[idx], y_all[idx]
if len(X_all) > 1000000: X_all, y_all = X_all[:1000000], y_all[:1000000]
print(f"Loaded {len(X_all)} samples, {X_all.shape[1]} dims, in {time.time()-t0:.1f}s")

class M(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

kf = KFold(n_splits=K, shuffle=True, random_state=42)
fold_accs = []; fold_histories = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    print(f"\n{'='*50}\nFOLD {fold_idx+1}/{K}\n{'='*50}")
    X_tr, X_val = X_all[train_idx], X_all[val_idx]; y_tr, y_val = y_all[train_idx], y_all[val_idx]
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0).clip(1e-6)
    X_tr = (X_tr-mean)/std; X_val = (X_val-mean)/std
    print(f"Train: {len(X_tr)}, Val: {len(X_val)}")

    model = M(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    Xt, yt = torch.tensor(X_tr, device=DEVICE), torch.tensor(y_tr, device=DEVICE)
    Xv, yv = torch.tensor(X_val, device=DEVICE), torch.tensor(y_val, device=DEVICE)

    if fold_idx == 0:
        BS = auto_tune_batch(model, X_tr, y_tr)
        print(f"  Using BS={BS}")

    best_acc = 0; hist = {'loss':[], 'acc':[]}
    for epoch in range(1, EPOCHS+1):
        model.train(); perm = torch.randperm(len(Xt)); tl, nb = 0, 0
        for i in range(0, len(Xt), BS):
            bi = perm[i:i+BS]; opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(model(Xt[bi]), yt[bi])
            l.backward(); opt.step(); tl += l.item(); nb += 1
        hist['loss'].append(tl/nb)
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(Xv)) > 0.5).float()
            acc = (pred == yv).float().mean().item()
            hist['acc'].append(acc)
            if acc > best_acc: best_acc = acc
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={hist['loss'][-1]:.4f} acc={acc:.4f}")
    fold_accs.append(best_acc); fold_histories.append({'fold': fold_idx+1, 'loss': hist['loss'], 'acc': hist['acc']})
    print(f"  Best: {best_acc:.4f}")

print(f"\n{'='*50}\nK-FOLD (k={K})\n{'='*50}")
print(f"Folds: {[f'{a:.4f}' for a in fold_accs]}")
print(f"Mean: {np.mean(fold_accs):.4f}  Std: {np.std(fold_accs):.4f}")

# Final model
mean_all, std_all = X_all.mean(axis=0), X_all.std(axis=0).clip(1e-6)
Xa = (X_all - mean_all) / std_all
fm = M(Xa.shape[1]).to(DEVICE)
fm = torch.compile(fm, dynamic=True)
opt = torch.optim.Adam(fm.parameters(), lr=0.001, weight_decay=1e-5)
Xat = torch.tensor(Xa, device=DEVICE); yat = torch.tensor(y_all, device=DEVICE)
final_hist = {'loss': []}
for epoch in range(1, EPOCHS+1):
    fm.train(); perm = torch.randperm(len(Xa)); tl, nb = 0, 0
    for i in range(0, len(Xa), BS):
        bi = perm[i:i+BS]; opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(fm(Xat[bi]), yat[bi])
        l.backward(); opt.step(); tl += l.item(); nb += 1
    final_hist['loss'].append(tl/nb)

# Save for inference (PyTorch)
torch.save({"model": fm.state_dict(), "mean": mean_all.tolist(), "std": std_all.tolist()},
           MODEL_DIR / "edge_classifier.pt")

mlp_log = {"model": "MLP Edge Classifier (GPU)", "k": K, "epochs": EPOCHS,
           "fold_accuracies": [float(a) for a in fold_accs],
           "mean": float(np.mean(fold_accs)), "std": float(np.std(fold_accs)),
           "fold_histories": fold_histories, "final": final_hist}
with open(MODEL_DIR / "mlp_train_log.json", "w") as f: json.dump(mlp_log, f, indent=2)
print(f"Total time: {time.time()-t0:.0f}s")
print(f"Model saved: {os.path.getsize(MODEL_DIR / 'edge_classifier.pt')/1024:.0f} KB")
