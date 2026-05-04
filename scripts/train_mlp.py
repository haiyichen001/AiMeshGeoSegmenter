"""
MLP Edge Classifier: load pre-computed edge data, k-fold CV, training plots
"""
import os, random, time, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import KFold

ROOT = Path(r"D:\AiMeshGeoSegmenter")
MLP_DIR = ROOT / "data" / "mlp_edges"
MODEL_DIR = ROOT / "models"
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cpu")
K = 5; EPOCHS = 30; BS = 4096

files = sorted(f for f in os.listdir(MLP_DIR) if f.endswith('.npz'))
random.seed(42); random.shuffle(files)
print(f"Files: {len(files)}")

kf = KFold(n_splits=K, shuffle=True, random_state=42)
fold_accs = []; fold_histories = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(files)):
    print(f"\n{'='*50}\nFOLD {fold_idx+1}/{K}\n{'='*50}")
    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Load training data
    pos, neg = [], []
    for i, fn in enumerate(train_files):
        d = np.load(MLP_DIR / fn)
        X, y = d['X'], d['y']
        mask_pos = y == 1; mask_neg = y == 0
        n = min(mask_pos.sum(), mask_neg.sum())
        if n == 0: continue
        pidx = np.random.choice(np.where(mask_pos)[0], n, replace=False)
        nidx = np.random.choice(np.where(mask_neg)[0], n, replace=False)
        pos.append(X[pidx]); neg.append(X[nidx])
        if (i+1) % 4000 == 0: print(f"  [{i+1}/{len(train_files)}]")
    X_train = np.vstack(pos + neg)
    y_train = np.array([1]*sum(p.shape[0] for p in pos) + [0]*sum(n.shape[0] for n in neg), dtype=np.float32)
    idx = np.random.permutation(len(X_train)); X_train, y_train = X_train[idx], y_train[idx]
    mean, std = X_train.mean(axis=0), X_train.std(axis=0).clip(1e-6)
    X_train = (X_train - mean) / std

    # Validation
    vp, vn = [], []
    for fn in val_files:
        d = np.load(MLP_DIR / fn)
        X, y = d['X'], d['y']
        mp, mn = y==1, y==0
        nv = min(mp.sum(), mn.sum())
        if nv == 0: continue
        pi = np.random.choice(np.where(mp)[0], nv, replace=False)
        ni = np.random.choice(np.where(mn)[0], nv, replace=False)
        vp.append(X[pi]); vn.append(X[ni])
    X_val = np.vstack(vp+vn)
    y_val = np.array([1]*sum(p.shape[0] for p in vp)+[0]*sum(n.shape[0] for n in vn), dtype=np.float32)
    X_val = (X_val - mean) / std

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(5,64), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
        def forward(self, x): return self.net(x).squeeze(-1)

    model = M().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    Xt, yt = torch.tensor(X_train, device=DEVICE), torch.tensor(y_train, device=DEVICE)
    Xv, yv = torch.tensor(X_val, device=DEVICE), torch.tensor(y_val, device=DEVICE)

    best_acc = 0; hist = {'loss':[], 'acc':[]}
    for epoch in range(1, EPOCHS+1):
        model.train()
        perm = torch.randperm(len(Xt)); tl, nb = 0, 0
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
    fold_accs.append(best_acc)
    print(f"  Best val acc: {best_acc:.4f}")

    # Save fold history as JSON for frontend
    fold_histories.append({'fold': fold_idx+1, 'loss': hist['loss'], 'acc': hist['acc']})

print(f"\n{'='*50}\nK-FOLD (k={K})\n{'='*50}")
print(f"Folds: {[f'{a:.4f}' for a in fold_accs]}")
print(f"Mean: {np.mean(fold_accs):.4f}  Std: {np.std(fold_accs):.4f}")

# Save training log for frontend
import json as _json
mlp_log = {
    "model": "MLP Edge Classifier",
    "k": K, "epochs": EPOCHS,
    "fold_accuracies": [float(a) for a in fold_accs],
    "mean": float(np.mean(fold_accs)),
    "std": float(np.std(fold_accs)),
    "fold_histories": fold_histories,
    "final": final_hist,
}
with open(MODEL_DIR / "mlp_train_log.json", "w") as f:
    _json.dump(mlp_log, f, indent=2)
print(f"Log saved to models/mlp_train_log.json")

# Final model on all data
print(f"\nTraining final model on all {len(files)} files...")
all_pos, all_neg = [], []
for fn in files:
    d = np.load(MLP_DIR / fn); X, y = d['X'], d['y']
    mp, mn = y==1, y==0; n = min(mp.sum(), mn.sum())
    if n == 0: continue
    all_pos.append(X[np.random.choice(np.where(mp)[0], n, replace=False)])
    all_neg.append(X[np.random.choice(np.where(mn)[0], n, replace=False)])
Xa = np.vstack(all_pos+all_neg)
ya = np.array([1]*sum(p.shape[0] for p in all_pos)+[0]*sum(n.shape[0] for n in all_neg), dtype=np.float32)
idx = np.random.permutation(len(Xa)); Xa, ya = Xa[idx], ya[idx]
ma, sa = Xa.mean(axis=0), Xa.std(axis=0).clip(1e-6); Xa = (Xa - ma) / sa

fm = M().to(DEVICE); opt = torch.optim.Adam(fm.parameters(), lr=0.001, weight_decay=1e-5)
Xat, yat = torch.tensor(Xa, device=DEVICE), torch.tensor(ya, device=DEVICE)
final_hist = {'loss': []}
for epoch in range(1, EPOCHS+1):
    fm.train(); perm = torch.randperm(len(Xa)); tl, nb = 0, 0
    for i in range(0, len(Xa), BS): bi=perm[i:i+BS]; opt.zero_grad(); l=F.binary_cross_entropy_with_logits(fm(Xat[bi]), yat[bi]); l.backward(); opt.step(); tl += l.item(); nb += 1
    final_hist['loss'].append(tl/nb)

torch.save({"model": fm.state_dict(), "mean": ma, "std": sa}, MODEL_DIR / "edge_classifier.pt")
print(f"Model saved: {os.path.getsize(MODEL_DIR / 'edge_classifier.pt')/1024:.0f} KB")
