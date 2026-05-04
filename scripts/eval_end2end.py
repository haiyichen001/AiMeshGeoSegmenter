"""
端到端评估: 模型一次加载, 逐三角匹配
"""
import os, sys, json, time, random, numpy as np
from pathlib import Path
from scipy.spatial import KDTree

ROOT = Path(r"D:\AiMeshGeoSegmenter")
sys.path.insert(0, str(ROOT))

ANSWERS = ROOT / "answers"
STL_DIR = ROOT / "data" / "stl"

# Load model once
from scripts.infer import segment_mesh, patch_features, predict_stl

ans_files = sorted(ANSWERS.glob("*.npz"))[:100]
print(f"Evaluating {len(ans_files)} parts...")

accs_tri, accs_face = [], []
t0 = time.time()

for i, af in enumerate(ans_files):
    stem = af.stem
    stl = STL_DIR / f"{stem}.stl"
    if not stl.exists(): continue

    # Run inference
    pred = predict_stl(str(stl))

    # Load answer
    d = np.load(af)
    gt_verts = d["vertices"]
    gt_tris = d["triangles"]
    gt_fids = d["tri_face_ids"]
    gt_ftypes = d["face_types"]
    label_names = d["label_names"]

    # For each answer face, find its type
    n_gt_faces = len(gt_ftypes)

    # Build predicted triangle centers from patches
    pred_centers = []
    pred_types = []
    for patch in pred["faces"]:
        verts = patch["vertices"]
        tris = patch["triangles"]
        if len(verts) < 9 or len(tris) < 3: continue
        v = np.array(verts).reshape(-1, 3)
        t = np.array(tris).reshape(-1, 3)
        centers = v[t].mean(axis=1)
        pred_centers.append(centers)
        pred_types.extend([patch["type"]] * len(t))

    if not pred_centers: continue
    pred_centers = np.vstack(pred_centers)
    pred_types = np.array(pred_types)

    # Per-answer-triangle: find nearest predicted triangle
    gt_centers = gt_verts[gt_tris].mean(axis=1)
    tree = KDTree(pred_centers)
    _, idx = tree.query(gt_centers)
    matched_types = pred_types[idx]

    # Per-triangle accuracy
    max_fid = len(gt_ftypes) - 1
    gt_tri_types = []
    for ti in range(len(gt_fids)):
        fid = min(gt_fids[ti], max_fid)
        gt_tri_types.append(str(label_names[gt_ftypes[fid]]))
    gt_tri_types = np.array(gt_tri_types)
    tri_acc = (matched_types == gt_tri_types).mean()
    accs_tri.append(tri_acc)

    # Per-face accuracy: majority vote per face
    face_correct = 0
    for fi in range(n_gt_faces):
        mask = gt_fids == fi
        if mask.sum() == 0: continue
        face_tri_matched = matched_types[mask]
        gt_type = str(label_names[gt_ftypes[fi]])
        # Majority vote
        pred_type = max(set(face_tri_matched), key=list(face_tri_matched).count)
        if pred_type == gt_type: face_correct += 1
    accs_face.append(face_correct / max(n_gt_faces, 1))

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(ans_files)} tri={np.mean(accs_tri):.3f} face={np.mean(accs_face):.3f}")

elapsed = time.time() - t0
print(f"\nDone: {len(accs_tri)} parts in {elapsed:.0f}s ({elapsed/max(len(accs_tri),1):.1f}s/part)")
print(f"Per-triangle accuracy: {np.mean(accs_tri):.4f} +/- {np.std(accs_tri):.4f}")
print(f"Per-face accuracy:     {np.mean(accs_face):.4f} +/- {np.std(accs_face):.4f}")
