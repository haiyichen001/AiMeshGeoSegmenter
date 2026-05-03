"""
MLP Edge Classifier: train from STL data with k-fold CV + training plots

Key improvement: train on actual STL triangle pairs (not STEP tessellation)
to eliminate train/inference domain mismatch.
"""
import os, sys, json, random, time, collections
from pathlib import Path
import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r"D:\AiMeshGeoSegmenter")
STL_DIR = ROOT / "data" / "stl"
LABEL_DIR = ROOT / "data" / "labels"
MODEL_DIR = ROOT / "models"
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cpu")
K_FOLDS = 5
EPOCHS = 30
BS = 4096


def process_one(label_file):
    """Extract edge training data from one STL+label pair."""
    stem = label_file.replace('.json', '')
    stl_path = STL_DIR / f"{stem}.stl"
    label_path = LABEL_DIR / label_file

    if not stl_path.exists():
        return [], []

    try:
        # Load STL
        mesh = trimesh.load(str(stl_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        stl_verts = np.array(mesh.vertices, dtype=np.float32)
        stl_faces = np.array(mesh.faces, dtype=np.int32)

        # Load label JSON
        with open(label_path) as f:
            label_data = json.load(f)

        # Build face-ID mapping for STL triangles via center proximity
        face_centers = []
        face_ids_stl = []
        for face in label_data["faces"]:
            verts = face.get("vertices", [])
            tris = face.get("triangles", [])
            if len(verts) < 9 or len(tris) < 3:
                continue
            v = np.array(verts).reshape(-1, 3)
            t = np.array(tris).reshape(-1, 3)
            centers = v[t].mean(axis=1)
            face_centers.append(centers)
            face_ids_stl.append(face["id"])

        if not face_centers:
            return [], []

        all_centers = np.vstack(face_centers)
        all_ids = np.concatenate([[fid] * len(c) for fid, c in zip(face_ids_stl, face_centers)])

        from scipy.spatial import KDTree
        tree = KDTree(all_centers)
        stl_centers = stl_verts[stl_faces].mean(axis=1)
        _, idx = tree.query(stl_centers)
        stl_face_ids = all_ids[idx]

        # STL triangle adjacency
        adj = mesh.face_adjacency
        if len(adj) == 0:
            return [], []

        # Features per edge
        tri_centers = stl_centers
        e1 = stl_verts[stl_faces[:, 1]] - stl_verts[stl_faces[:, 0]]
        e2 = stl_verts[stl_faces[:, 2]] - stl_verts[stl_faces[:, 0]]
        tri_normals = np.cross(e1, e2)
        tri_nrm = np.linalg.norm(tri_normals, axis=1, keepdims=True).clip(1e-15)
        tri_areas = tri_nrm.flatten() * 0.5
        tri_normals /= tri_nrm

        positives, negatives = [], []
        for a, b in adj:
            dihedral = np.arccos(np.clip(np.dot(tri_normals[a], tri_normals[b]), -1, 1)) * 180 / np.pi
            area_a, area_b = tri_areas[a], tri_areas[b]
            area_ratio = min(area_a, area_b) / max(max(area_a, area_b), 1e-12)
            # Shared edge ratio
            shared = set(stl_faces[a]) & set(stl_faces[b])
            edge_len = 0.001
            if len(shared) >= 2:
                s = list(shared)[:2]
                edge_len = np.linalg.norm(stl_verts[s[0]] - stl_verts[s[1]])
            avg_side = np.sqrt(max(area_a, 1e-12) * 2 / np.sqrt(3))
            edge_ratio = edge_len / max(avg_side, 1e-6)

            feat = [dihedral, dihedral, area_ratio, edge_ratio, 0.0]
            if stl_face_ids[a] == stl_face_ids[b]:
                positives.append(feat)
            else:
                negatives.append(feat)

        return positives, negatives
    except:
        return [], []


if __name__ == "__main__":
    files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json') and f != 'distribution.json']
    random.seed(42); random.shuffle(files)
    print(f"Parts: {len(files)}")

    n_workers = max(1, cpu_count() - 1)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_accs = []
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(files)):
        print(f"\n{'='*50}\nFOLD {fold_idx+1}/{K_FOLDS}\n{'='*50}")
        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        print(f"Train: {len(train_files)}, Val: {len(val_files)}")

        # Extract training data
        all_pos, all_neg = [], []
        t0 = time.time()
        with Pool(n_workers) as p:
            for i, (pos, neg) in enumerate(p.imap_unordered(process_one, train_files, chunksize=10)):
                all_pos.extend(pos); all_neg.extend(neg)
                if (i+1) % 2000 == 0:
                    print(f"  [{i+1}/{len(train_files)}] pos={len(all_pos)} neg={len(all_neg)}")
        print(f"  Extracted in {time.time()-t0:.0f}s: pos={len(all_pos)} neg={len(all_neg)}")

        n_train = min(len(all_pos), len(all_neg))
        pos_train = random.sample(all_pos, n_train)
        neg_train = random.sample(all_neg, n_train)
        X = np.array(pos_train + neg_train, dtype=np.float32)
        y = np.array([1]*n_train + [0]*n_train, dtype=np.float32)
        idx = np.random.permutation(len(X)); X, y = X[idx], y[idx]
        mean, std = X.mean(axis=0), X.std(axis=0).clip(1e-6)
        X = (X - mean) / std

        # Validation data
        val_pos, val_neg = [], []
        with Pool(n_workers) as p:
            for pos, neg in p.imap_unordered(process_one, val_files, chunksize=10):
                val_pos.extend(pos); val_neg.extend(neg)
        n_val = min(len(val_pos), len(val_neg))
        vp = random.sample(val_pos, n_val); vn = random.sample(val_neg, n_val)
        X_val = np.array(vp+vn, dtype=np.float32)
        y_val = np.array([1]*n_val+[0]*n_val, dtype=np.float32)
        X_val = (X_val - mean) / std

        # Model
        class EdgeCls(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(5,64), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(64,32), nn.ReLU(),
                                         nn.Linear(32,1))
            def forward(self, x):
                return self.net(x).squeeze(-1)

        model = EdgeCls().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        Xt, yt = torch.tensor(X, device=DEVICE), torch.tensor(y, device=DEVICE)
        Xv, yv = torch.tensor(X_val, device=DEVICE), torch.tensor(y_val, device=DEVICE)

        best_acc = 0
        hist = {'train_loss': [], 'val_acc': []}
        for epoch in range(1, EPOCHS+1):
            model.train()
            perm = torch.randperm(len(X))
            total_loss = 0; n_batches = 0
            for i in range(0, len(X), BS):
                bi = perm[i:i+BS]; opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model(Xt[bi]), yt[bi])
                loss.backward(); opt.step()
                total_loss += loss.item(); n_batches += 1
            hist['train_loss'].append(total_loss/n_batches)

            model.eval()
            with torch.no_grad():
                pred = (torch.sigmoid(model(Xv)) > 0.5).float()
                acc = (pred == yv).float().mean().item()
                hist['val_acc'].append(acc)
                if acc > best_acc: best_acc = acc

        fold_accs.append(best_acc)
        fold_losses.append(hist)
        print(f"  Best val acc: {best_acc:.4f}")

        # Save fold plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(hist['train_loss']); ax1.set_title(f'Fold {fold_idx+1} Train Loss'); ax1.set_xlabel('Epoch')
        ax2.plot(hist['val_acc']); ax2.set_title(f'Fold {fold_idx+1} Val Acc'); ax2.set_xlabel('Epoch')
        plt.tight_layout(); plt.savefig(MODEL_DIR / f'mlp_fold{fold_idx+1}.png', dpi=100); plt.close()

    # Summary
    print(f"\n{'='*50}\nK-FOLD RESULTS (k={K_FOLDS})\n{'='*50}")
    print(f"Fold accuracies: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean: {np.mean(fold_accs):.4f}  Std: {np.std(fold_accs):.4f}")

    # Summary plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(range(1, K_FOLDS+1), fold_accs, color='steelblue')
    ax.axhline(np.mean(fold_accs), color='red', linestyle='--', label=f'Mean={np.mean(fold_accs):.4f}')
    ax.set_xticks(range(1, K_FOLDS+1))
    ax.set_ylabel('Validation Accuracy'); ax.set_xlabel('Fold'); ax.legend()
    ax.set_ylim(0.9, 1.0)
    plt.tight_layout(); plt.savefig(MODEL_DIR / 'mlp_kfold_summary.png', dpi=100); plt.close()

    # Train final model on all data
    print(f"\nTraining final model on all {len(files)} parts...")
    all_pos, all_neg = [], []
    with Pool(n_workers) as p:
        for pos, neg in p.imap_unordered(process_one, files, chunksize=10):
            all_pos.extend(pos); all_neg.extend(neg)
    n_all = min(len(all_pos), len(all_neg))
    pa = random.sample(all_pos, n_all); na = random.sample(all_neg, n_all)
    X_all = np.array(pa+na, dtype=np.float32)
    y_all = np.array([1]*n_all+[0]*n_all, dtype=np.float32)
    idx = np.random.permutation(len(X_all)); X_all, y_all = X_all[idx], y_all[idx]
    mean_all, std_all = X_all.mean(axis=0), X_all.std(axis=0).clip(1e-6)
    X_all = (X_all - mean_all) / std_all

    final_model = EdgeCls().to(DEVICE)
    opt = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)
    Xa, ya = torch.tensor(X_all, device=DEVICE), torch.tensor(y_all, device=DEVICE)
    hist_final = {'loss': []}
    for epoch in range(1, EPOCHS+1):
        final_model.train()
        perm = torch.randperm(len(X_all)); tl = 0; nb = 0
        for i in range(0, len(X_all), BS):
            bi = perm[i:i+BS]; opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(final_model(Xa[bi]), ya[bi])
            l.backward(); opt.step(); tl += l.item(); nb += 1
        hist_final['loss'].append(tl/nb)

    torch.save({"model": final_model.state_dict(), "mean": mean_all, "std": std_all},
               MODEL_DIR / "edge_classifier.pt")
    print(f"Final model saved: {os.path.getsize(MODEL_DIR / 'edge_classifier.pt')/1024:.0f} KB")

    # Final training plot
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(hist_final['loss']); ax.set_title('Final Model - Training Loss')
    ax.set_xlabel('Epoch'); plt.tight_layout()
    plt.savefig(MODEL_DIR / 'mlp_final_train.png', dpi=100); plt.close()
    print(f"Plots saved to {MODEL_DIR}/")
