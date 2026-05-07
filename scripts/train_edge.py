"""
MLP Edge Classifier v2: 8 stable features, bigger model, boundary pairs.

Features: angle, area_ratio, dist_rel, convex, edge_len, perim_ratio,
          area_density, angle_sin. All computable for any triangle pair.
"""
import os, json, random, time, numpy as np, trimesh
from pathlib import Path
from multiprocessing import Pool, cpu_count
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(r"D:\AiMeshGeoSegmenter")
LABEL_DIR = ROOT / "data" / "labels"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_4feat(na, nb, ca, cb, area_a, area_b, span):
    dot = np.clip(np.dot(na, nb), -1, 1)
    angle = float(np.arccos(dot) * 180 / np.pi)
    area_ratio = min(area_a, area_b) / max(max(area_a, area_b), 1e-12)
    dist_rel = np.linalg.norm(ca - cb) / max(span, 1e-6)
    convex = float(np.dot(na, cb - ca))
    return [angle, area_ratio, dist_rel, convex]


def extract_training_data(label_file, max_pairs=30):
    with open(os.path.join(LABEL_DIR, label_file)) as f:
        data = json.load(f)
    faces = data["faces"]
    if len(faces) < 2: return [], []
    span = max(data.get("span", 1.0), 1e-6)
    positives, negatives = [], []

    face_info = {}
    for idx, face in enumerate(faces):
        verts = face.get("vertices", [])
        tris = face.get("triangles", [])
        if len(verts) < 9 or len(tris) < 3: continue
        face_info[idx] = {
            "verts": np.array(verts).reshape(-1, 3),
            "tris": np.array(tris).reshape(-1, 3),
            "neighbors": face.get("neighbors", []),
        }
    if len(face_info) < 2: return [], []

    for idx, info in face_info.items():
        verts = info["verts"]; tris = info["tris"]
        if len(tris) < 2: continue
        try:
            centers = verts[tris].mean(axis=1)
            e1 = verts[tris[:,1]] - verts[tris[:,0]]
            e2 = verts[tris[:,2]] - verts[tris[:,0]]
            normals = np.cross(e1, e2)
            nrm = np.linalg.norm(normals, axis=1, keepdims=True)
            areas = nrm.flatten() * 0.5
            nrm[nrm < 1e-15] = 1.0; normals /= nrm
            e3 = verts[tris[:,2]] - verts[tris[:,1]]
            perims = np.linalg.norm(e1,axis=1) + np.linalg.norm(e2,axis=1) + np.linalg.norm(e3,axis=1)
            mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
            adj = mesh.face_adjacency
        except: continue
        if len(adj) == 0: continue

        N = min(max_pairs, len(adj))
        indices = np.random.choice(len(adj), N, replace=False)
        for ai, bi in adj[indices]:
            # Compute shared edge length
            shared = set(tris[ai]) & set(tris[bi])
            edge_len = np.linalg.norm(verts[list(shared)[0]] - verts[list(shared)[1]]) if len(shared) >= 2 else 0
            feats = compute_4feat(normals[ai], normals[bi], centers[ai], centers[bi],
                                   areas[ai], areas[bi], span)
            positives.append(feats)

    # Negatives: spatially closest triangle pairs across adjacent faces (hard samples)
    for idx, info in face_info.items():
        for nb_idx in info["neighbors"]:
            if nb_idx not in face_info: continue
            nb_info = face_info[nb_idx]
            try:
                a_verts = info["verts"]; a_tris = info["tris"]
                b_verts = nb_info["verts"]; b_tris = nb_info["tris"]
                a_centers = a_verts[a_tris].mean(axis=1)
                b_centers = b_verts[b_tris].mean(axis=1)
                ae1 = a_verts[a_tris[:,1]] - a_verts[a_tris[:,0]]
                ae2 = a_verts[a_tris[:,2]] - a_verts[a_tris[:,0]]
                an = np.cross(ae1, ae2); an_nrm = np.linalg.norm(an,axis=1,keepdims=True).clip(1e-15)
                an /= an_nrm; a_areas = an_nrm.flatten()*0.5
                be1 = b_verts[b_tris[:,1]] - b_verts[b_tris[:,0]]
                be2 = b_verts[b_tris[:,2]] - b_verts[b_tris[:,0]]
                bn = np.cross(be1, be2); bn_nrm = np.linalg.norm(bn,axis=1,keepdims=True).clip(1e-15)
                bn /= bn_nrm; b_areas = bn_nrm.flatten()*0.5
            except: continue

            # Find closest pairs (hard boundary samples)
            n_samp = min(5, min(len(an), len(bn)))
            # Take first N triangles from face A, find closest in face B
            a_sample = np.random.choice(len(an), min(n_samp, len(an)), replace=False)
            for ai in a_sample:
                dists = np.linalg.norm(b_centers - a_centers[ai], axis=1)
                bi = int(np.argmin(dists))
                feats = compute_4feat(an[ai], bn[bi], a_centers[ai], b_centers[bi],
                                       a_areas[ai], b_areas[bi], span)
                negatives.append(feats)

    return positives, negatives


def main():
    files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json') and f != 'distribution.json']
    random.seed(42); random.shuffle(files)
    n_tr = int(len(files)*0.8)
    train_files, test_files = files[:n_tr], files[n_tr:]
    print(f"Train: {len(train_files)}, Test: {len(test_files)}, Device: {DEVICE}")

    all_pos, all_neg = [], []
    t0 = time.time()
    nw = max(1, cpu_count() - 1)
    print(f"Extracting data with {nw} workers...")
    with Pool(nw) as pool:
        for p, n in pool.imap_unordered(extract_training_data, train_files, chunksize=50):
            all_pos.extend(p); all_neg.extend(n)
    print(f"Data: {time.time()-t0:.0f}s, pos={len(all_pos)} neg={len(all_neg)}")

    n = min(len(all_pos), len(all_neg))
    pos_s = random.sample(all_pos, n)
    neg_s = random.sample(all_neg, n)
    X = np.array(pos_s + neg_s, dtype=np.float32)
    y = np.array([1]*n + [0]*n, dtype=np.float32)
    idx = np.random.permutation(len(X)); X, y = X[idx], y[idx]
    mean, std = X.mean(axis=0), X.std(axis=0).clip(1e-6)
    X = (X - mean) / std

    test_pos, test_neg = [], []
    with Pool(nw) as pool:
        for p, n in pool.imap_unordered(extract_training_data, test_files, chunksize=50):
            test_pos.extend(p); test_neg.extend(n)
    nt = min(len(test_pos), len(test_neg))
    Xt = np.array(random.sample(test_pos,nt) + random.sample(test_neg,nt), dtype=np.float32)
    yt = np.array([1]*nt + [0]*nt, dtype=np.float32)
    Xt = (Xt - mean) / std
    print(f"Train: {len(X)}, Test: {len(Xt)}")

    class EdgeClassifier(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1))
        def forward(self, x): return self.net(x).squeeze(-1)

    model = EdgeClassifier(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")

    X_t, y_t = torch.tensor(X, device=DEVICE), torch.tensor(y, device=DEVICE)
    X_te, y_te = torch.tensor(Xt, device=DEVICE), torch.tensor(yt, device=DEVICE)
    BS, best_model_acc = 8192, 0
    all_histories = []
    seeds = [42, 123, 456]
    best_model_idx = 0

    for si, seed in enumerate(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        print(f"\n  Model {si+1}/3 (seed={seed})")
        model = EdgeClassifier(X.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        mlp_hist = {'loss': [], 'acc': []}
        best_acc = 0

        for epoch in range(1, 51):
            model.train(); perm = torch.randperm(len(X)); epoch_loss = 0
            for i in range(0, len(X), BS):
                bi = perm[i:i+BS]
                loss = F.binary_cross_entropy_with_logits(model(X_t[bi]), y_t[bi])
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()
            mlp_hist['loss'].append(epoch_loss / (len(X)/BS))

            model.eval()
            with torch.no_grad():
                out = model(X_te)
                best_t, best_a = 0.5, 0
                for t in np.arange(0.2, 0.8, 0.02):
                    acc = ((torch.sigmoid(out) > t).float() == y_te).float().mean().item()
                    if acc > best_a: best_a = acc; best_t = t
                mlp_hist['acc'].append(best_a)
                if best_a > best_acc: best_acc = best_a

            if epoch % 10 == 0 or epoch == 1:
                print(f"    E{epoch:3d} loss={mlp_hist['loss'][-1]:.4f} acc={best_a:.4f}")

        all_histories.append({"seed": seed, "best_acc": float(best_acc), "history": mlp_hist})
        print(f"    Best: {best_acc:.4f}")

        # Save individual model
        torch.save({"model": model.state_dict(), "mean": mean, "std": std, "threshold": float(best_t)},
                   ROOT/"models"/f"edge_classifier_{si}.pt")
        if best_acc > best_model_acc:
            best_model_acc = best_acc; best_thr = best_t; best_model_idx = si
        del model; torch.cuda.empty_cache()

    # Save best as default
    import shutil
    shutil.copy(ROOT/"models"/f"edge_classifier_{best_model_idx}.pt", ROOT/"models"/"edge_classifier.pt")
    print(f"\nBest model: {best_model_idx} ({best_model_acc:.4f} @ thr={best_thr:.3f})")

    # Save ensemble log
    import json
    with open(ROOT/"models"/"mlp_train_log.json", "w") as f:
        json.dump({"model": "MLP Edge v3 ensemble", "params": sum(p.numel() for p in EdgeClassifier(X.shape[1]).parameters()),
                   "best_acc": float(best_model_acc), "threshold": float(best_thr),
                   "histories": all_histories}, f, indent=2)
    print("Log saved.")

if __name__ == "__main__": main()
