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
    def __init__(self, in_dim=10, hidden=128, heads=4, n_classes=8, n_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        ch = [in_dim] + [hidden*heads]*n_layers
        for i in range(n_layers):
            oh = hidden if i < n_layers-1 else hidden//2
            h = heads if i < n_layers-1 else 1
            self.convs.append(GATConv(ch[i], oh, heads=h, dropout=dropout))
            if i < n_layers-1: self.norms.append(nn.BatchNorm1d(oh*h))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden//2,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,n_classes))
    def forward(self, data):
        x,ei = data.x, data.edge_index; ei,_ = add_self_loops(ei, num_nodes=x.size(0))
        for i,conv in enumerate(self.convs):
            x = conv(x,ei)
            if i < len(self.norms): x = self.norms[i](x); x = F.elu(x); x = self.dropout(x)
        return F.log_softmax(self.mlp(x), dim=-1)


def predict_stl(stl_path, model_path=None):
    model_path = model_path or str(ROOT / "models" / "model.pt")
    model = TriangleGAT(in_dim=10)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()

    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene): mesh = trimesh.util.concatenate(mesh.dump())
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Features (matching training)
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
    if len(adj)>0:
        for a,b in adj: d=np.linalg.norm(normals[a]-normals[b]); nbr_norm_var[a]+=d; nbr_norm_var[b]+=d
    x = np.stack([normals[:,0],normals[:,1],normals[:,2],rel_ctr[:,0],rel_ctr[:,1],rel_ctr[:,2],
                  area_log,center_dist,nbr_norm_var,np.zeros(len(areas),dtype=np.float32)],axis=1)
    ei = adj.T if len(adj)>0 else np.zeros((2,1),dtype=np.int64)
    data = Data(x=torch.tensor(x), edge_index=torch.tensor(ei, dtype=torch.long))

    with torch.no_grad():
        pred = model(data).argmax(dim=1).numpy()

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
