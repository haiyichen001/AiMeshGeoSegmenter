"""
训练边分类器: 判断两个相邻三角形该合并还是切割

训练数据: 从现存 labels JSON 提取
  - 正样本 (merge=1): 同一 B-Rep 面内相邻三角形对
  - 负样本 (merge=0): 相邻 B-Rep 面的边界三角形对

模型: 轻量 MLP (5 维输入 → 二分类)
"""
import os, json, random, time, collections
from pathlib import Path
import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"D:\AiMeshGeoSegmenter")
LABEL_DIR = ROOT / "data" / "labels"

DEVICE = torch.device("cpu")


def extract_training_data(label_file, max_pairs_per_face=50):
    """从单个零件提取正负样本对."""
    with open(os.path.join(LABEL_DIR, label_file)) as f:
        data = json.load(f)

    faces = data["faces"]
    if len(faces) < 2:
        return [], []

    span = max(data.get("span", 1.0), 1e-6)
    positives = []  # within same face
    negatives = []  # across adjacent faces

    # Build face_id -> index map
    face_info = {}
    for idx, face in enumerate(faces):
        verts = face.get("vertices", [])
        tris = face.get("triangles", [])
        if len(verts) < 9 or len(tris) < 3:
            continue
        face_info[idx] = {
            "verts": np.array(verts).reshape(-1, 3),
            "tris": np.array(tris).reshape(-1, 3),
            "neighbors": face.get("neighbors", []),
        }

    if len(face_info) < 2:
        return [], []

    # Within-face positives
    for idx, info in face_info.items():
        verts = info["verts"]
        tris = info["tris"]
        if len(tris) < 2:
            continue

        try:
            # Compute triangle centers and normals
            tri_centers = verts[tris].mean(axis=1)  # (N, 3)
            e1 = verts[tris[:, 1]] - verts[tris[:, 0]]
            e2 = verts[tris[:, 2]] - verts[tris[:, 0]]
            tri_normals = np.cross(e1, e2)
            tri_nrm = np.linalg.norm(tri_normals, axis=1, keepdims=True)
            tri_areas = tri_nrm.flatten() * 0.5
            tri_nrm[tri_nrm < 1e-15] = 1.0
            tri_normals /= tri_nrm

            # Build triangle adjacency via trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
            adj = mesh.face_adjacency  # (M, 2)
        except:
            continue

        if len(adj) == 0:
            continue

        # Random sample adjacent pairs
        indices = np.random.choice(len(adj), min(max_pairs_per_face, len(adj)), replace=False)
        for a_idx, b_idx in adj[indices]:
            na = tri_normals[a_idx]; nb = tri_normals[b_idx]
            ca = tri_centers[a_idx]; cb = tri_centers[b_idx]
            dot = np.clip(np.dot(na, nb), -1, 1)
            angle = float(np.arccos(dot) * 180 / np.pi)
            area_ratio = min(tri_areas[a_idx], tri_areas[b_idx]) / max(max(tri_areas[a_idx], tri_areas[b_idx]), 1e-12)
            dist_rel = np.linalg.norm(ca - cb) / span
            dist_abs = np.linalg.norm(ca - cb)
            shape_a = float(np.sqrt(max(tri_areas[a_idx], 1e-12)) / max(np.linalg.norm(verts[tris[a_idx]] - verts[tris[a_idx]].mean(axis=0)).sum(), 1e-12))
            shape_b = float(np.sqrt(max(tri_areas[b_idx], 1e-12)) / max(np.linalg.norm(verts[tris[b_idx]] - verts[tris[b_idx]].mean(axis=0)).sum(), 1e-12))
            convex = float(np.dot(na, cb - ca))  # signed, indicates convex/concave

            positives.append([angle, area_ratio, dist_rel, convex])

    # Across-face negatives (triangles from adjacent B-Rep faces)
    for idx, info in face_info.items():
        for nb_idx in info["neighbors"]:
            if nb_idx not in face_info:
                continue
            nb_info = face_info[nb_idx]
            other_verts = nb_info["verts"]
            other_tris = nb_info["tris"]
            if len(other_tris) < 1 or len(info["tris"]) < 1:
                continue

            try:
                nc = tri_centers = info["verts"][info["tris"]].mean(axis=1)
                no = nb_info["verts"][nb_info["tris"]].mean(axis=1)

                e1 = info["verts"][info["tris"][:, 1]] - info["verts"][info["tris"][:, 0]]
                e2 = info["verts"][info["tris"][:, 2]] - info["verts"][info["tris"][:, 0]]
                na = np.cross(e1, e2)
                nrm_a = np.linalg.norm(na, axis=1, keepdims=True).clip(1e-15)
                na /= nrm_a

                e1 = nb_info["verts"][nb_info["tris"][:, 1]] - nb_info["verts"][nb_info["tris"][:, 0]]
                e2 = nb_info["verts"][nb_info["tris"][:, 2]] - nb_info["verts"][nb_info["tris"][:, 0]]
                nb = np.cross(e1, e2)
                nrm_b = np.linalg.norm(nb, axis=1, keepdims=True).clip(1e-15)
                nb /= nrm_b
            except:
                continue

            # Take random triangle pairs from the two faces
            n_pairs = min(5, min(len(na), len(nb)))
            for _ in range(n_pairs):
                ai = np.random.randint(len(na))
                bi = np.random.randint(len(nb))
                dot = np.clip(np.dot(na[ai], nb[bi]), -1, 1)
                angle = float(np.arccos(dot) * 180 / np.pi)
                dist_rel = np.linalg.norm(nc[ai] - no[bi]) / span
                dist_abs = np.linalg.norm(nc[ai] - no[bi])
                area_a = np.linalg.norm(np.cross(
                    info["verts"][info["tris"][ai,1]] - info["verts"][info["tris"][ai,0]],
                    info["verts"][info["tris"][ai,2]] - info["verts"][info["tris"][ai,0]]))
                area_b = np.linalg.norm(np.cross(
                    nb_info["verts"][nb_info["tris"][bi,1]] - nb_info["verts"][nb_info["tris"][bi,0]],
                    nb_info["verts"][nb_info["tris"][bi,2]] - nb_info["verts"][nb_info["tris"][bi,0]]))
                area_ratio = min(area_a, area_b) / max(max(area_a, area_b), 1e-12)
                shape_a = 0.5; shape_b = 0.5  # approximate for boundary pairs
                convex = float(np.dot(na[ai], no[bi] - nc[ai]))
                negatives.append([angle, area_ratio, dist_rel, convex])

    return positives, negatives


def main():
    # Load all label files
    files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json') and f != 'distribution.json']
    random.seed(42)
    random.shuffle(files)

    # Split
    n_train = int(len(files) * 0.8)
    train_files = files[:n_train]
    test_files = files[n_train:]

    print(f"Train parts: {len(train_files)}, Test parts: {len(test_files)}")

    # Extract training data
    all_pos, all_neg = [], []
    t0 = time.time()
    for i, fn in enumerate(train_files):
        pos, neg = extract_training_data(fn, max_pairs_per_face=20)
        all_pos.extend(pos)
        all_neg.extend(neg)
        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{len(train_files)}] pos={len(all_pos)} neg={len(all_neg)}")

    print(f"Extracted in {time.time()-t0:.0f}s: pos={len(all_pos)} neg={len(all_neg)}")

    # Balance classes
    n_samples = min(len(all_pos), len(all_neg))
    pos = random.sample(all_pos, n_samples)
    neg = random.sample(all_neg, n_samples)

    X = np.array(pos + neg, dtype=np.float32)
    y = np.array([1] * n_samples + [0] * n_samples, dtype=np.float32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Normalize features
    mean = X.mean(axis=0)
    std = X.std(axis=0).clip(1e-6)
    X = (X - mean) / std

    # Test data
    test_pos, test_neg = [], []
    for fn in test_files:
        pos, neg = extract_training_data(fn, max_pairs_per_face=10)
        test_pos.extend(pos)
        test_neg.extend(neg)

    n_test = min(len(test_pos), len(test_neg))
    test_pos = random.sample(test_pos, n_test)
    test_neg = random.sample(test_neg, n_test)
    X_test = np.array(test_pos + test_neg, dtype=np.float32)
    y_test = np.array([1] * n_test + [0] * n_test, dtype=np.float32)
    X_test = (X_test - mean) / std

    print(f"Train: {len(X)}, Test: {len(X_test)}")

    # Model: tiny MLP
    class EdgeClassifier(nn.Module):
        def __init__(self, in_dim=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = EdgeClassifier(X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")

    # Training
    X_t = torch.tensor(X, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)
    X_te = torch.tensor(X_test, device=DEVICE)
    y_te = torch.tensor(y_test, device=DEVICE)

    batch_size = 4096
    best_acc = 0

    for epoch in range(1, 41):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = perm[i:i + batch_size]
            out = model(X_t[batch_idx])
            loss = F.binary_cross_entropy_with_logits(out, y_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(X_te)
            pred = (torch.sigmoid(out) > 0.5).float()
            acc = (pred == y_te).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                torch.save({"model": model.state_dict(), "mean": mean, "std": std},
                           ROOT / "models" / "edge_classifier.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | test acc={acc:.4f}")

    print(f"\nBest test accuracy: {best_acc:.4f}")

    # Model size
    size_kb = os.path.getsize(ROOT / "models" / "edge_classifier.pt") / 1024
    print(f"Model saved: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
