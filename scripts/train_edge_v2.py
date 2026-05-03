"""
边分类器 v2: 从 STEP 生成统一三角网格，提取真实边界训练数据

每个三角形标记所属 B-Rep 面 ID → 相邻三角形同面=merge, 不同面=cut
"""
import os, sys, json, random, time, collections, tempfile, shutil
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count

ROOT = Path(r"D:\AiMeshGeoSegmenter")
STEP_DIR = ROOT / "data" / "step"

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


def extract_edges_from_step(step_path):
    """Tessellate STEP, return unified mesh + per-triangle face IDs + adjacency."""
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        return None, None, None, None
    reader.TransferRoots()
    shape = reader.OneShape()

    bbox = Bnd_Box(); brepbndlib.Add(shape, bbox)
    x1,y1,z1,x2,y2,z2 = bbox.Get()
    span = max(x2-x1, y2-y1, z2-z1)

    BRepMesh_IncrementalMesh(shape, span / 100.0).Perform()

    all_verts = []
    all_tris = []
    all_face_ids = []  # which B-Rep face each triangle belongs to

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0
    while exp.More():
        face = exp.Current()
        loc = TopLoc_Location()
        tri = BRep_Tool().Triangulation(face, loc)
        if tri is None:
            exp.Next(); continue

        trsf = loc.Transformation()
        nv = tri.NbNodes()
        nt = tri.NbTriangles()

        local_idx = {}
        for i in range(1, nv + 1):
            p = tri.Node(i); p.Transform(trsf)
            all_verts.append([p.X(), p.Y(), p.Z()])
            local_idx[i] = len(all_verts) - 1

        for i in range(1, nt + 1):
            t = tri.Triangle(i)
            all_tris.append([local_idx[t.Value(1)], local_idx[t.Value(2)], local_idx[t.Value(3)]])
            all_face_ids.append(face_idx)

        face_idx += 1
        exp.Next()

    if len(all_tris) < 2:
        return None, None, None, None

    all_verts = np.array(all_verts, dtype=np.float32)
    all_tris = np.array(all_tris, dtype=np.int32)
    all_face_ids = np.array(all_face_ids, dtype=np.int32)

    # Deduplicate vertices
    uniq, inv = np.unique(all_verts, axis=0, return_inverse=True)
    all_tris_flat = inv[all_tris.flatten()].reshape(-1, 3)

    # Build edge adjacency using trimesh
    import trimesh
    mesh = trimesh.Trimesh(vertices=uniq, faces=all_tris_flat, process=False)
    adj = mesh.face_adjacency  # (M, 2)

    return uniq, all_tris_flat, all_face_ids, adj


def process_one(step_file):
    """Extract edge training data from one STEP file."""
    path = STEP_DIR / step_file
    try:
        verts, tris, face_ids, adj = extract_edges_from_step(path)
    except:
        return [], []

    if adj is None or len(adj) == 0:
        return [], []

    span = max(verts.max(axis=0) - verts.min(axis=0))
    if span < 1e-6: span = 1.0

    # Triangle centers and normals
    tri_centers = verts[tris].mean(axis=1)
    e1 = verts[tris[:, 1]] - verts[tris[:, 0]]
    e2 = verts[tris[:, 2]] - verts[tris[:, 0]]
    tri_normals = np.cross(e1, e2)
    tri_nrm = np.linalg.norm(tri_normals, axis=1, keepdims=True).clip(1e-15)
    tri_areas = tri_nrm.flatten() * 0.5
    tri_normals /= tri_nrm

    # Dihedral angles at shared edges
    dihedral = []
    for a, b in adj:
        ea = tris[a]
        eb = tris[b]
        shared = set(ea) & set(eb)
        if len(shared) < 2:
            dihedral.append(0.0)
            continue
        s = list(shared)[:2]
        edge_vec = verts[s[1]] - verts[s[0]]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-10:
            dihedral.append(0.0)
            continue
        # Dihedral = angle between the two triangle planes
        dot = np.clip(np.dot(tri_normals[a], tri_normals[b]), -1, 1)
        dihedral.append(np.arccos(dot) * 180 / np.pi)

    dihedral = np.array(dihedral)

    # Build samples
    positives, negatives = [], []
    area_a = tri_areas[adj[:, 0]]
    area_b = tri_areas[adj[:, 1]]
    area_min = np.minimum(area_a, area_b)
    area_max = np.maximum(area_a, area_b)
    area_ratio = np.where(area_max > 1e-12, area_min / area_max, 1.0)

    # Edge length / avg triangle perimeter ratio
    edge_lengths = []
    for a, b in adj:
        ea = tris[a]; eb = tris[b]
        shared = list(set(ea) & set(eb))
        if len(shared) >= 2:
            edge_lengths.append(np.linalg.norm(verts[shared[0]] - verts[shared[1]]))
        else:
            edge_lengths.append(0.001)
    edge_lengths = np.array(edge_lengths)
    avg_side = np.sqrt(area_a * 2 / np.sqrt(3))  # approximate side length from area
    avg_side = np.maximum(avg_side, 1e-6)
    edge_ratio = edge_lengths / avg_side

    normal_angle = dihedral  # same thing in this context

    for i in range(len(adj)):
        feat = [
            dihedral[i],                    # 0: dihedral angle at shared edge
            normal_angle[i],                # 1: normal angle (same as dihedral for adjacent tris)
            area_ratio[i],                  # 2: area ratio
            edge_ratio[i],                  # 3: shared edge / avg side ratio
            0.0,                            # 4: placeholder
        ]
        if face_ids[adj[i, 0]] == face_ids[adj[i, 1]]:
            positives.append(feat)
        else:
            negatives.append(feat)

    return positives, negatives


if __name__ == "__main__":
    files = [f for f in os.listdir(STEP_DIR) if f.endswith('.step')]
    random.seed(42)
    random.shuffle(files)

    n_train = int(len(files) * 0.8)
    train_files = files[:n_train]
    test_files = files[n_train:]

    print(f"Train parts: {len(train_files)}, Test parts: {len(test_files)}")

    n_workers = max(1, cpu_count() - 1)
    print(f"Workers: {n_workers}")

    all_pos, all_neg = [], []
    t0 = time.time()

    with Pool(n_workers) as p:
        for i, (pos, neg) in enumerate(p.imap_unordered(process_one, train_files, chunksize=10)):
            all_pos.extend(pos)
            all_neg.extend(neg)
            if (i + 1) % 2000 == 0:
                print(f"  [{i+1}/{len(train_files)}] pos={len(all_pos)} neg={len(all_neg)}")

    print(f"Extracted in {time.time()-t0:.0f}s: pos={len(all_pos)} neg={len(all_neg)}")

    # Balance
    n_samples = min(len(all_pos), len(all_neg))
    pos = random.sample(all_pos, n_samples)
    neg = random.sample(all_neg, n_samples)

    X = np.array(pos + neg, dtype=np.float32)
    y = np.array([1] * n_samples + [0] * n_samples, dtype=np.float32)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    mean = X.mean(axis=0)
    std = X.std(axis=0).clip(1e-6)
    X = (X - mean) / std

    # Test
    test_pos, test_neg = [], []
    with Pool(n_workers) as p:
        for pos, neg in p.imap_unordered(process_one, test_files, chunksize=10):
            test_pos.extend(pos)
            test_neg.extend(neg)

    n_test = min(len(test_pos), len(test_neg))
    test_pos = random.sample(test_pos, n_test)
    test_neg = random.sample(test_neg, n_test)
    X_test = np.array(test_pos + test_neg, dtype=np.float32)
    y_test = np.array([1] * n_test + [0] * n_test, dtype=np.float32)
    X_test = (X_test - mean) / std

    print(f"Train: {len(X)}, Test: {len(X_test)}")

    # Model
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class EdgeCls(nn.Module):
        def __init__(self, in_dim=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = EdgeCls(5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    X_te = torch.tensor(X_test)
    y_te = torch.tensor(y_test)

    best_acc = 0
    bs = 4096
    for epoch in range(1, 41):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            bi = perm[i:i+bs]
            out = model(X_t[bi])
            loss = F.binary_cross_entropy_with_logits(out, y_t[bi])
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
    print(f"Model saved: {os.path.getsize(ROOT / 'models' / 'edge_classifier.pt') / 1024:.0f} KB")
