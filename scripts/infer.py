"""
GAT 推理: STL -> per-triangle 6 类标签
"""
import sys, json, numpy as np, trimesh
from pathlib import Path

ROOT = Path(__file__).parent.parent
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv

LABEL_NAMES = ["plane","cylinder","sphere","cone","torus","freeform"]
COLORS = {"plane":"#4db8ff","cylinder":"#44cc44","sphere":"#ff44ff","cone":"#ff8844",
          "torus":"#ffcc00","freeform":"#888888"}

class TriangleGAT(nn.Module):
    def __init__(self, in_dim=36, hidden=192, heads=4, n_classes=6, n_layers=3, dropout=0.3, edge_dim=3):
        super().__init__()
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        ch = [in_dim] + [hidden*heads]*n_layers
        for i in range(n_layers):
            oh = hidden if i < n_layers-1 else hidden//2
            h = heads if i < n_layers-1 else 1
            self.convs.append(GATConv(ch[i], oh, heads=h, dropout=dropout, edge_dim=edge_dim))
            if i < n_layers-1: self.norms.append(nn.BatchNorm1d(oh*h))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden//2,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,n_classes))
    def forward(self, data):
        x,ei,ea = data.x, data.edge_index, data.edge_attr
        ei,_ = add_self_loops(ei, num_nodes=x.size(0))
        n_self = x.size(0)
        ea_full = torch.cat([ea, torch.zeros(n_self, ea.size(1), device=x.device)], dim=0)
        for i,conv in enumerate(self.convs):
            x = conv(x, ei, ea_full)
            if i < len(self.norms): x = self.norms[i](x); x = F.elu(x); x = self.dropout(x)
        return F.log_softmax(self.mlp(x), dim=-1)


def merge_regions(faces, normals, pred_labels, angle_deg=15):
    """Region growing + majority vote: clean per-triangle predictions.

    1. Merge adjacent triangles into regions based on normal angle < angle_deg.
    2. Within each region, majority vote on predicted labels -> assign to all.
    Returns cleaned per-triangle label array.
    """
    import trimesh
    m = trimesh.Trimesh(vertices=np.zeros((faces.max()+1, 3), dtype=np.float32),
                        faces=faces, process=False)
    adj = m.face_adjacency

    # Build adjacency list
    nbrs = [[] for _ in range(len(faces))]
    for a, b in adj:
        nbrs[a].append(b)
        nbrs[b].append(a)

    visited = np.zeros(len(faces), dtype=bool)
    regions = []  # list of (triangle_indices, label_counts)

    for seed in range(len(faces)):
        if visited[seed]:
            continue
        # Region growing via BFS
        region = [seed]
        visited[seed] = True
        head = 0
        while head < len(region):
            ti = region[head]; head += 1
            for nb in nbrs[ti]:
                if visited[nb]:
                    continue
                # Check normal angle between current and neighbor
                dot = min(1.0, max(0.0, abs(np.dot(normals[ti], normals[nb]))))
                ang = float(np.arccos(dot) * 180 / np.pi)
                if ang < angle_deg:
                    visited[nb] = True
                    region.append(nb)

        # Majority vote within region
        labels_in_region = pred_labels[region]
        counts = np.bincount(labels_in_region, minlength=8)
        majority = int(np.argmax(counts))
        regions.append((region, majority))

    # Assign cleaned labels
    cleaned = pred_labels.copy()
    for region, label in regions:
        cleaned[region] = label

    return cleaned


def predict_stl(stl_path, model_paths=None, refine=True):
    if model_paths is None:
        model_paths = [str(ROOT / "models" / "model.pt")]

    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene): mesh = trimesh.util.concatenate(mesh.dump())
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Features (matching training: 14 dim + edge_attr)
    e1 = verts[faces[:,1]]-verts[faces[:,0]]; e2 = verts[faces[:,2]]-verts[faces[:,0]]
    normals = np.cross(e1,e2); nrm = np.linalg.norm(normals,axis=1,keepdims=True).clip(1e-15)
    normals /= nrm; areas = nrm.flatten()*0.5
    centers = verts[faces].mean(axis=1)
    part_ctr = centers.mean(axis=0); part_r = float(np.sqrt(((centers-part_ctr)**2).sum(axis=1).mean()))
    area_log = np.log10(np.maximum(areas,1e-6))
    center_dist = np.sqrt(((centers-part_ctr)**2).sum(axis=1))/max(part_r,1e-6)
    rel_ctr = (centers-part_ctr)/max(part_r,1e-6)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    adj = m.face_adjacency

    nbr_norm_var = np.zeros(len(areas), dtype=np.float32)
    dihedral = np.zeros(len(areas), dtype=np.float32)
    max_dih = np.zeros(len(areas), dtype=np.float32)
    e3 = verts[faces[:,2]] - verts[faces[:,1]]
    l1 = np.linalg.norm(e1, axis=1); l2 = np.linalg.norm(e2, axis=1); l3 = np.linalg.norm(e3, axis=1)
    perimeter = l1 + l2 + l3
    edge_ratio = np.minimum(np.minimum(l1,l2),l3) / np.maximum(np.maximum(l1,l2),np.maximum(l3,1e-12))
    vn_std = np.zeros(len(areas), dtype=np.float32)

    if len(adj)>0:
        nbr_cnt = np.zeros(len(areas), dtype=np.int32)
        for a,b in adj:
            d=np.linalg.norm(normals[a]-normals[b]); nbr_norm_var[a]+=d; nbr_norm_var[b]+=d
            vn_std[a] += d; vn_std[b] += d
            dot = np.clip(np.abs(np.dot(normals[a], normals[b])), 0, 1)
            ang = float(np.arccos(dot) * 180 / np.pi)
            dihedral[a] += ang; dihedral[b] += ang
            max_dih[a] = max(max_dih[a], ang); max_dih[b] = max(max_dih[b], ang)
            nbr_cnt[a] += 1; nbr_cnt[b] += 1
        mask = nbr_cnt > 0; dihedral[mask] /= nbr_cnt[mask]

    shape = np.sqrt(np.maximum(areas, 1e-12)) / np.maximum(perimeter * 0.07, 1e-12)

    x = np.stack([normals[:,0],normals[:,1],normals[:,2], rel_ctr[:,0],rel_ctr[:,1],rel_ctr[:,2],
                  area_log,center_dist,nbr_norm_var, dihedral, max_dih, shape, edge_ratio, vn_std], axis=1)
    ei = adj.T if len(adj)>0 else np.zeros((2,1),dtype=np.int64)

    # Edge features
    edge_attr = np.zeros((len(adj), 3), dtype=np.float32) if len(adj)>0 else np.zeros((0,3), dtype=np.float32)
    if len(adj) > 0:
        for ei_idx, (a,b) in enumerate(adj):
            ca, cb = centers[a], centers[b]
            dot_ab = np.clip(np.abs(np.dot(normals[a], normals[b])), 0, 1)
            edge_attr[ei_idx,0] = float(np.arccos(dot_ab))
            edge_attr[ei_idx,1] = np.linalg.norm(ca-cb) / max((perimeter[a]+perimeter[b])/2.0, 1e-6)
            edge_attr[ei_idx,2] = float(np.dot(normals[a], cb-ca))
    ea = torch.tensor(edge_attr, dtype=torch.float32) if len(adj)>0 else torch.zeros(0, 3)

    # Ensemble: average predictions from all models
    all_log_probs = []
    for mp in model_paths:
        model = TriangleGAT(in_dim=14)
        model.load_state_dict(torch.load(mp, map_location='cpu', weights_only=False))
        model.eval()
        data = Data(x=torch.tensor(x), edge_index=torch.tensor(ei, dtype=torch.long), edge_attr=ea)
        with torch.no_grad():
            log_prob = model(data).exp().numpy()  # softmax -> probabilities
        all_log_probs.append(log_prob)

    avg_prob = np.mean(all_log_probs, axis=0)
    pred = avg_prob.argmax(axis=1)

    # Post-processing: region growing + voting
    if refine:
        pred = merge_regions(faces, normals, pred)

    # Group by predicted label -> patches for visualization
    faces_out = []
    for li, name in enumerate(LABEL_NAMES):
        mask = pred == li
        if mask.sum() == 0: continue
        tris = faces[mask]
        # Build vertex set for this label
        vset = {}; vi = 0; loc_verts = []; loc_tris = []
        for t in tris:
            lt = []
            for v in t:
                if v not in vset: vset[v]=vi; loc_verts.extend(verts[v].tolist()); vi+=1
                lt.append(vset[v])
            loc_tris.extend(lt)
        cx=sum(loc_verts[0::3])/vi; cy=sum(loc_verts[1::3])/vi; cz=sum(loc_verts[2::3])/vi
        faces_out.append({"vertices":loc_verts,"triangles":loc_tris,"type":name,"color":COLORS[name],"center":[cx,cy,cz]})

    span = float(max(verts.max(axis=0)-verts.min(axis=0)))
    return {"faces":faces_out,"center":centers.mean(axis=0).tolist(),"span":span,"num_patches":len(faces_out)}


if __name__ == "__main__":
    if len(sys.argv) < 2: print(json.dumps({"error":"usage: python infer.py <stl_path>"})); sys.exit(1)
    print(json.dumps(predict_stl(sys.argv[1])))
