"""
STL → 面类型预测 (segmentation + GNN)

Pipeline:
  STL → region-growing分割 → 每patch提18维特征 → GNN预测 → 8类标签+可视化
"""
import numpy as np
import trimesh
from collections import defaultdict
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data

# 8-class
LABELS = ["plane", "cylinder", "sphere", "cone", "torus", "fillet", "chamfer", "freeform"]
COLORS = {
    "plane": "#4db8ff", "cylinder": "#44cc44", "sphere": "#ff44ff",
    "cone": "#ff8844", "torus": "#ffcc00", "fillet": "#00cccc",
    "chamfer": "#ff6644", "freeform": "#888888",
}


def segment_mesh(vertices, faces, angle_thresh_deg=20.0):
    """Region-growing: merge adjacent triangles with similar normals."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    face_normals = mesh.face_normals
    adjacency = mesh.face_adjacency  # (N, 2) pairs

    # Build adjacency list
    neighbors = defaultdict(list)
    for a, b in adjacency:
        neighbors[a].append(b)
        neighbors[b].append(a)

    n_faces = len(faces)
    labels = -np.ones(n_faces, dtype=int)
    cos_thresh = np.cos(np.radians(angle_thresh_deg))
    current_label = 0

    for seed in range(n_faces):
        if labels[seed] >= 0:
            continue
        labels[seed] = current_label
        queue = [seed]
        ref_normal = face_normals[seed]
        while queue:
            fi = queue.pop()
            for nb in neighbors.get(fi, []):
                if labels[nb] >= 0:
                    continue
                if np.dot(ref_normal, face_normals[nb]) > cos_thresh:
                    labels[nb] = current_label
                    queue.append(nb)
        current_label += 1

    # Merge small patches (< 5 triangles) into largest neighbor
    patch_counts = np.bincount(labels)
    tiny = set(np.where(patch_counts < 5)[0])
    for fi in range(n_faces):
        pid = labels[fi]
        if pid not in tiny:
            continue
        best_pid = pid
        for nb in neighbors.get(fi, []):
            if labels[nb] not in tiny:
                best_pid = labels[nb]
                break
        labels[fi] = best_pid

    # Re-map to consecutive IDs
    unique = sorted(set(labels))
    new_id = {old: i for i, old in enumerate(unique)}
    labels = np.array([new_id[l] for l in labels], dtype=int)
    return labels


def patch_features(vertices, faces, patch_labels):
    """Extract 18-dim features per patch (same as training)."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    n_patches = patch_labels.max() + 1
    face_areas = mesh.area_faces
    face_normals = mesh.face_normals
    part_center = vertices.mean(axis=0)
    part_span = max(vertices.max(axis=0) - vertices.min(axis=0))
    if part_span < 1e-6:
        part_span = 1.0

    # Per-patch stats
    patch_faces = [[] for _ in range(n_patches)]
    for fi, pid in enumerate(patch_labels):
        patch_faces[pid].append(fi)

    total_area = face_areas.sum()

    # Build adjacency between patches
    adj = mesh.face_adjacency
    patch_adj = defaultdict(set)
    for a, b in adj:
        pa, pb = patch_labels[a], patch_labels[b]
        if pa != pb:
            patch_adj[pa].add(pb)
            patch_adj[pb].add(pa)

    features = np.zeros((n_patches, 18), dtype=np.float32)
    edge_index = [[], []]

    for pid in range(n_patches):
        faces_p = patch_faces[pid]
        if not faces_p:
            continue

        area = face_areas[faces_p].sum()
        normals = face_normals[faces_p]
        mean_normal = normals.mean(axis=0)
        nr = np.linalg.norm(mean_normal)
        if nr > 1e-15:
            mean_normal /= nr
        normal_std = normals.std(axis=0).mean()

        # Center
        centers = vertices[faces[faces_p]].mean(axis=1)
        patch_center = centers.mean(axis=0)
        rel_center = (patch_center - part_center) / max(part_span, 1e-6)

        # Neighbors
        nbrs = sorted(patch_adj.get(pid, set()))
        n_nbrs = len(nbrs)

        # Dihedral
        dihedral = []
        for nb in nbrs:
            if patch_faces[nb]:
                nb_normal = face_normals[patch_faces[nb]].mean(axis=0)
                nbr_nr = np.linalg.norm(nb_normal)
                if nbr_nr > 1e-15:
                    nb_normal /= nbr_nr
                dot = abs(np.dot(mean_normal, nb_normal))
                dot = min(1.0, max(0.0, dot))
                dihedral.append(np.arccos(dot) * 180 / np.pi)
        dih_mean = np.mean(dihedral) if dihedral else 0.0
        dih_std = np.std(dihedral) if dihedral else 0.0

        n_tris = len(faces_p)
        area_log = np.log10(max(area, 1e-6))
        area_ratio = area / max(total_area, 1e-6)
        n_tris_log = np.log10(max(n_tris, 1))
        vert_density_log = np.log10(max(n_tris * 3 / max(area, 1e-6), 1e-6))
        n_verts_log = np.log10(max(n_tris * 3, 1))

        features[pid] = [
            area_log, mean_normal[0], mean_normal[1], mean_normal[2],
            normal_std, rel_center[0], rel_center[1], rel_center[2],
            n_tris_log, float(n_nbrs), 1.0, 1.0, 1.0,
            vert_density_log, area_ratio, dih_mean, dih_std, n_verts_log,
        ]

        for nb in nbrs:
            edge_index[0].append(pid)
            edge_index[1].append(nb)

    return features, np.array(edge_index, dtype=np.int64), patch_faces


def predict_stl(stl_path, model_path=None):
    """Full STL inference pipeline. Returns dict ready for frontend."""
    # Load STL
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Segment
    patch_ids = segment_mesh(vertices, faces)

    # Features
    feats, edge_index, patch_faces = patch_features(vertices, faces, patch_ids)

    # GNN predict
    class GNN(nn.Module):
        def __init__(self, in_d, hid=128, n_cls=8, n_layers=3, drop=0.3):
            super().__init__()
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.convs.append(SAGEConv(in_d, hid))
            self.norms.append(nn.BatchNorm1d(hid))
            for _ in range(n_layers - 1):
                self.convs.append(SAGEConv(hid, hid))
                self.norms.append(nn.BatchNorm1d(hid))
            self.drop = nn.Dropout(drop)
            self.mlp = nn.Sequential(
                nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(hid // 2, n_cls),
            )
        def forward(self, data):
            x, ei = data.x, data.edge_index
            ei, _ = add_self_loops(ei, num_nodes=x.size(0))
            for conv, norm in zip(self.convs, self.norms):
                x = conv(x, ei); x = norm(x); x = F.relu(x); x = self.drop(x)
            return F.log_softmax(self.mlp(x), dim=-1)

    from pathlib import Path as P
    model = GNN(feats.shape[1])
    model_path = model_path or str(P(__file__).parent.parent / "models" / "face_classifier.pt")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        x = torch.tensor(feats)
        ei = torch.tensor(edge_index)
        data = Data(x=x, edge_index=ei, num_nodes=feats.shape[0])
        out = model(data)
        pred = out.argmax(dim=1).numpy()

    # Build output
    face_labels = pred[patch_ids]  # map patch -> per-face

    # Per-patch faces for rendering
    patch_data = []
    for pid in range(len(patch_faces)):
        if not patch_faces[pid]:
            continue
        label_name = LABELS[pred[pid]]
        # Collect triangle vertices for this patch
        triangle_list = []
        for fi in patch_faces[pid]:
            triangle_list.extend([int(x) for x in faces[fi]])
        # Deduplicate and remap
        verts_flat = []
        tris_flat = []
        vmap = {}
        vi = 0
        for fi in patch_faces[pid]:
            tri = []
            for v in faces[fi]:
                if v not in vmap:
                    vmap[v] = vi
                    verts_flat.extend([float(vertices[v][0]), float(vertices[v][1]), float(vertices[v][2])])
                    vi += 1
                tri.append(vmap[v])
            tris_flat.extend(tri)

        cx = sum(verts_flat[0::3]) / vi if vi > 0 else 0
        cy = sum(verts_flat[1::3]) / vi if vi > 0 else 0
        cz = sum(verts_flat[2::3]) / vi if vi > 0 else 0

        patch_data.append({
            "vertices": verts_flat,
            "triangles": tris_flat,
            "type": label_name,
            "color": COLORS[label_name],
            "center": [cx, cy, cz],
        })

    center = vertices.mean(axis=0).tolist()
    span = float(max(vertices.max(axis=0) - vertices.min(axis=0)))

    return {
        "faces": patch_data,
        "center": center,
        "span": span,
        "num_patches": len(patch_data),
    }


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: python infer.py <stl_path>"}))
        sys.exit(1)
    result = predict_stl(sys.argv[1])
    print(json.dumps(result))
    sys.exit(0)
