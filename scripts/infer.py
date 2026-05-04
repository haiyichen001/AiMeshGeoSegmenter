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
from pathlib import Path as P
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


def segment_mesh(vertices, faces):
    """AI edge classifier: predict merge/cut for each adjacent triangle pair."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center
    face_areas = mesh.area_faces
    adjacency = mesh.face_adjacency
    n_faces = len(faces)
    span = max(vertices.max(axis=0) - vertices.min(axis=0))
    if span < 1e-6:
        span = 1.0

    # Load PyTorch edge classifier
    ckpt_path = str(P(__file__).parent.parent / "models" / "edge_classifier.pt")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    edge_model = EdgeMLP(8).eval()
    edge_model.load_state_dict(ckpt["model"])
    mean = torch.tensor(ckpt["mean"])
    std = torch.tensor(ckpt["std"])

    # Edge classifier: sklearn SGD on 8-dim features
    edges_keep = []
    if len(adjacency) > 0:
        na = face_normals[adjacency[:, 0]]
        nb = face_normals[adjacency[:, 1]]
        dihedral = np.arccos(np.clip((na * nb).sum(axis=1), -1, 1)) * 180 / np.pi
        area_min = np.minimum(face_areas[adjacency[:, 0]], face_areas[adjacency[:, 1]])
        area_max = np.maximum(face_areas[adjacency[:, 0]], face_areas[adjacency[:, 1]])
        area_ratio = np.where(area_max > 1e-12, area_min / area_max, 1.0)

        edge_len_ratio = np.zeros(len(adjacency))
        aspect_tri = np.zeros(len(adjacency)); max_angle_tri = np.zeros(len(adjacency))
        min_aspect = np.zeros(len(adjacency))
        for idx, (a, b) in enumerate(adjacency):
            shared = list(set(faces[a]) & set(faces[b]))
            if len(shared) >= 2:
                el = np.linalg.norm(vertices[shared[0]] - vertices[shared[1]])
                avg_s = np.sqrt(max(face_areas[a], 1e-12) * 2 / np.sqrt(3))
                edge_len_ratio[idx] = el / max(avg_s, 1e-6)
            # Triangle shape
            def tri_shape(vi):
                pts = vertices[faces[vi]]
                sd = sorted([np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[1]-pts[2]), np.linalg.norm(pts[2]-pts[0])])
                asp = sd[2]/max(sd[0], 1e-6)
                cos_max = (sd[0]**2+sd[1]**2-sd[2]**2)/max(2*sd[0]*sd[1], 1e-12)
                max_ang = np.arccos(np.clip(cos_max, -1, 1))*180/np.pi
                return asp, max_ang
            asp_a, ang_a = tri_shape(a); asp_b, ang_b = tri_shape(b)
            aspect_tri[idx] = max(asp_a, asp_b)
            max_angle_tri[idx] = max(ang_a, ang_b)
            min_aspect[idx] = min(asp_a, asp_b)

        X = np.stack([dihedral, dihedral, area_ratio, edge_len_ratio, np.zeros(len(adjacency)),
                      aspect_tri, max_angle_tri, min_aspect], axis=1)
        X_t = (torch.tensor(X, dtype=torch.float32) - mean) / std
        with torch.no_grad():
            logits = edge_model(X_t)
            keep = (torch.sigmoid(logits) > 0.5).numpy()

        for i in range(len(adjacency)):
            if keep[i]:
                edges_keep.append((adjacency[i, 0], adjacency[i, 1]))

    # Connected components
    neighbors = defaultdict(list)
    for a, b in edges_keep:
        neighbors[a].append(b)
        neighbors[b].append(a)

    labels = -np.ones(n_faces, dtype=int)
    current_label = 0
    for seed in range(n_faces):
        if labels[seed] >= 0:
            continue
        labels[seed] = current_label
        queue = [seed]
        while queue:
            fi = queue.pop()
            for nb in neighbors.get(fi, []):
                if labels[nb] >= 0:
                    continue
                labels[nb] = current_label
                queue.append(nb)
        current_label += 1

    # Assign isolated faces to nearest patch
    for fi in range(n_faces):
        if labels[fi] >= 0:
            continue
        # Find closest adjacent face with a label
        mesh_adj = mesh.face_adjacency
        found = False
        for a, b in mesh_adj:
            if a == fi and labels[b] >= 0:
                labels[fi] = labels[b]
                found = True
                break
            if b == fi and labels[a] >= 0:
                labels[fi] = labels[a]
                found = True
                break
        if not found:
            labels[fi] = current_label
            current_label += 1

    # Re-map to consecutive IDs
    unique = sorted(set(labels))
    new_id = {old: i for i, old in enumerate(unique)}
    labels = np.array([new_id[l] for l in labels], dtype=int)
    return labels


class EdgeMLP(nn.Module):
    def __init__(self, in_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


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

    features = np.zeros((n_patches, 26), dtype=np.float32)
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

        # Curvature proxy: vertex normal deviation angles
        if n_tris >= 1:
            v_arr = vertices[faces[faces_p]]
            e1 = v_arr[:, 1] - v_arr[:, 0]
            e2 = v_arr[:, 2] - v_arr[:, 0]
            tri_n = np.cross(e1, e2)
            tri_nrm = np.linalg.norm(tri_n, axis=1, keepdims=True).clip(1e-15)
            tri_n /= tri_nrm
            angles = np.arccos(np.clip(np.dot(tri_n, mean_normal), -1, 1)) * 180 / np.pi
            na_mean = float(angles.mean()); na_std = float(angles.std())
            na_max = float(angles.max()); na_range = float(angles.max() - angles.min())
        else:
            na_mean = na_std = na_max = na_range = 0.0

        # Curvature features
        na_mean = na_std = na_max = na_range = 0.0
        cov_eig1 = cov_eig2 = cov_eig3 = cov_sum_log = 0.0
        if n_tris >= 2:
            fn_cov = np.cov(face_normals.T)
            eigvals = np.linalg.eigvalsh(fn_cov) if len(face_normals) >= 3 else np.zeros(3)
            if len(eigvals) >= 3:
                cov_eig1, cov_eig2, cov_eig3 = float(eigvals[2]), float(eigvals[1]), float(eigvals[0])
            cov_sum_log = np.log10(max(cov_eig1 + cov_eig2 + cov_eig3, 1e-12))
            angles = np.arccos(np.clip(np.dot(face_normals, mean_normal), -1, 1)) * 180 / np.pi
            na_mean = float(angles.mean()); na_std = float(angles.std())
            na_max = float(angles.max()); na_range = float(angles.max() - angles.min())

        features[pid] = [
            area_log, mean_normal[0], mean_normal[1], mean_normal[2],
            normal_std, rel_center[0], rel_center[1], rel_center[2],
            n_tris_log, float(n_nbrs), 1.0, 1.0, 1.0,
            vert_density_log, area_ratio, dih_mean, dih_std, n_verts_log,
            na_mean, na_std, na_max, na_range,
            cov_eig1, cov_eig2, cov_eig3, cov_sum_log,
        ]

        for nb in nbrs:
            edge_index[0].append(pid)
            edge_index[1].append(nb)

    return features, np.array(edge_index, dtype=np.int64), patch_faces


def predict_stl(stl_path, model_path=None):
    """Full STL inference pipeline. Returns dict ready for frontend."""
    import sys as _sys
    import pathlib
    _sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    # Load STL
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Segment
    patch_ids = segment_mesh(vertices, faces)

    # Features
    from scripts.extract_features_stl import compute_features_for_patches
    feats, edge_index = compute_features_for_patches(vertices, faces, patch_ids)
    # Build patch_faces from patch_ids
    patch_faces = [[] for _ in range(patch_ids.max()+1)]
    for fi, pid in enumerate(patch_ids): patch_faces[pid].append(fi)

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
            self.res_proj = nn.Linear(in_d, hid) if in_d != hid else nn.Identity()
            self.mlp = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(),
                                     nn.Dropout(drop), nn.Linear(hid//2, n_cls))
        def forward(self, data):
            x, ei = data.x, data.edge_index
            ei, _ = add_self_loops(ei, num_nodes=x.size(0))
            x0 = self.res_proj(x)
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                xn = conv(x, ei); xn = norm(xn); xn = F.relu(xn); xn = self.drop(xn)
                x = xn + (x0 if i < len(self.convs)-1 else xn*0)
            return F.log_softmax(self.mlp(x), dim=-1)

    from pathlib import Path as P
    model = GNN(feats.shape[1])
    model_path = model_path or str(P(__file__).parent.parent / "models" / "face_classifier.pt")
    model.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=False))
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
