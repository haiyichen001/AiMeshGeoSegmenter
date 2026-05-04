"""
MLP Edge Classifier: sklearn SGDClassifier for fast CPU training + k-fold CV
"""
import os, json, random, time, numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

ROOT = Path(r"D:\AiMeshGeoSegmenter")
MLP_DIR = ROOT / "data" / "mlp_edges"
MODEL_DIR = ROOT / "models"
os.makedirs(MODEL_DIR, exist_ok=True)

t0 = time.time()
files = sorted(f for f in os.listdir(MLP_DIR) if f.endswith('.npz'))
print(f"Loading {len(files)} files into memory...")
pos, neg = [], []
for i, fn in enumerate(files):
    d = np.load(MLP_DIR / fn); X, y = d['X'], d['y']
    mp, mn = y==1, y==0; n = min(mp.sum(), mn.sum())
    if n == 0: continue
    pi = np.random.choice(np.where(mp)[0], min(n, 100), replace=False)
    ni = np.where(mn)[0]; ni = ni[np.argsort(X[ni,0])[:min(n,100)]]
    pos.append(X[pi]); neg.append(X[ni])
    if (i+1) % 4000 == 0: print(f"  {i+1}/{len(files)}")
X_all = np.vstack(pos+neg).astype(np.float32)
y_all = np.array([1]*sum(len(p) for p in pos) + [0]*sum(len(n) for n in neg), dtype=np.float32)
idx = np.random.permutation(len(X_all)); X_all, y_all = X_all[idx], y_all[idx]
# Cap at 500K
if len(X_all) > 500000:
    X_all, y_all = X_all[:500000], y_all[:500000]
print(f"Loaded {len(X_all)} samples, {X_all.shape[1]} dims, in {time.time()-t0:.1f}s")

# Standardize
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)
fold_accs = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    X_tr, X_val = X_all[train_idx], X_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=100,
                        tol=1e-3, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_val, clf.predict(X_val))
    fold_accs.append(float(acc))
    print(f"Fold {fold_idx+1}: {acc:.4f}")

print(f"\nK-FOLD (k={K})")
print(f"Folds: {[f'{a:.4f}' for a in fold_accs]}")
print(f"Mean: {np.mean(fold_accs):.4f}  Std: {np.std(fold_accs):.4f}")

# Train final model
clf_final = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=200,
                          tol=1e-3, random_state=42, n_jobs=-1)
clf_final.fit(X_all, y_all)

# Save model + scaler
with open(MODEL_DIR / "edge_classifier_sklearn.pkl", "wb") as f:
    pickle.dump({"model": clf_final, "scaler": scaler}, f)

# Save log for dashboard
mlp_log = {
    "model": "MLP Edge Classifier (SGD)",
    "k": K,
    "fold_accuracies": fold_accs,
    "mean": float(np.mean(fold_accs)),
    "std": float(np.std(fold_accs)),
    "fold_histories": [{"fold": i+1, "loss": [], "acc": [a]} for i, a in enumerate(fold_accs)],
}
with open(MODEL_DIR / "mlp_train_log.json", "w") as f:
    json.dump(mlp_log, f, indent=2)

print(f"\nTotal time: {time.time()-t0:.0f}s")
print(f"Model saved to models/edge_classifier_sklearn.pkl")
