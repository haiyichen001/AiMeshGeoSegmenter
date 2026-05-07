"""
AiMeshGeoSegmenter Inference
Usage: python infer_standalone.py <input.stl>
Deps: pip install torch torch_geometric numpy trimesh
"""
import sys, json, numpy as np, trimesh
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

ROOT = Path(__file__).parent
LABEL_NAMES = ["plane","cylinder","sphere","cone","torus","freeform"]
COLORS = {"plane":"#4db8ff","cylinder":"#44cc44","sphere":"#ff44ff","cone":"#ff8844","torus":"#ffcc00","freeform":"#888888"}


class TriangleGAT(nn.Module):
    def __init__(self, in_dim=26, hidden=192, heads=4, n_classes=6, n_layers=3, dropout=0.3, edge_dim=3):
        super().__init__()
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        ch = [in_dim] + [hidden*heads]*n_layers
        for i in range(n_layers):
            oh = hidden if i < n_layers-1 else hidden//2
            h = heads if i < n_layers-1 else 1
            self.convs.append(GATConv(ch[i], oh, heads=h, dropout=dropout, edge_dim=edge_dim))
            if i < n_layers-1: self.norms.append(nn.BatchNorm1d(oh*h))
        self.dropout = nn.Dropout(dropout)
        jk_dim = in_dim
        for i in range(n_layers):
            h = heads if i < n_layers-1 else 1
            out = (hidden if i < n_layers-1 else hidden//2) * h
            jk_dim += out
        self.mlp = nn.Sequential(nn.Linear(jk_dim,128),nn.ReLU(),nn.Dropout(dropout),
                                 nn.Linear(128,64),nn.ReLU(),nn.Dropout(dropout),
                                 nn.Linear(64,n_classes))
    def forward(self, data):
        x,ei,ea = data.x, data.edge_index, data.edge_attr
        n_self = x.size(0)
        ei,_ = add_self_loops(ei, num_nodes=n_self)
        ea_full = torch.cat([ea, torch.zeros(n_self, ea.size(1), device=x.device)], dim=0)
        all_x = [x]
        for i,conv in enumerate(self.convs):
            x = conv(x, ei, ea_full)
            if i < len(self.norms): x = self.norms[i](x); x = F.elu(x); x = self.dropout(x)
            all_x.append(x)
        x_cat = torch.cat(all_x, dim=-1)
        return F.log_softmax(self.mlp(x_cat), dim=-1)


class EdgeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,64),nn.ReLU(),nn.Dropout(0.1),
                                 nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.1),
                                 nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self, x): return self.net(x).squeeze(-1)


def compute_features(verts, faces, normals, centers, areas, span):
    n = len(areas)
    verts_min = verts.min(axis=0); verts_max = verts.max(axis=0)
    part_extent = verts_max - verts_min
    part_diag = float(np.sqrt((part_extent**2).sum()))
    part_ctr = (verts_min + verts_max) / 2.0
    rel_ctr = (centers - part_ctr) / max(part_diag/2, 1e-6)
    fourier = []
    for freq in [np.pi, 2*np.pi]:
        for axis in range(3):
            fourier.append(np.sin(freq * rel_ctr[:, axis]))
            fourier.append(np.cos(freq * rel_ctr[:, axis]))
    e1 = verts[faces[:,1]] - verts[faces[:,0]]
    e2 = verts[faces[:,2]] - verts[faces[:,0]]
    e3 = verts[faces[:,2]] - verts[faces[:,1]]
    l1,l2,l3 = np.linalg.norm(e1,axis=1),np.linalg.norm(e2,axis=1),np.linalg.norm(e3,axis=1)
    perimeter = l1+l2+l3; shape = np.sqrt(np.maximum(areas,1e-12))/np.maximum(perimeter*0.07,1e-12)
    edge_ratio = np.minimum(np.minimum(l1,l2),l3)/np.maximum(np.maximum(l1,l2),np.maximum(l3,1e-12))
    elongation = part_extent / max(part_diag, 1e-6)
    x = np.stack([
        normals[:,0],normals[:,1],normals[:,2], *[fourier[i] for i in range(12)],
        np.log10(np.maximum(areas,1e-6)), np.zeros(n,np.float32), np.zeros(n,np.float32),
        np.zeros(n,np.float32), shape, edge_ratio, np.zeros(n,np.float32),
        np.full(n, np.log10(max(n,1)), np.float32), np.full(n, elongation[0], np.float32),
        np.full(n, elongation[1], np.float32), np.zeros(n,np.float32)
    ], axis=1)
    return x


def predict(stl_path):
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene): mesh = trimesh.util.concatenate(mesh.dump())
    verts = np.array(mesh.vertices, np.float32)
    faces = np.array(mesh.faces, np.int32)

    e1=verts[faces[:,1]]-verts[faces[:,0]]; e2=verts[faces[:,2]]-verts[faces[:,0]]
    normals=np.cross(e1,e2); nrm=np.linalg.norm(normals,axis=1,keepdims=True).clip(1e-15)
    normals/=nrm; areas=nrm.flatten()*0.5
    centers=verts[faces].mean(axis=1)
    span=float(max(verts.max(axis=0)-verts.min(axis=0)))

    m=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
    adj=m.face_adjacency
    nbr_norm_var=np.zeros(len(areas),np.float32)
    dihedral=np.zeros(len(areas),np.float32); max_dih=np.zeros(len(areas),np.float32)
    vn_std=np.zeros(len(areas),np.float32)
    if len(adj)>0:
        nbr_cnt=np.zeros(len(areas),np.int32)
        for a,b in adj:
            d=np.linalg.norm(normals[a]-normals[b]); nbr_norm_var[a]+=d; nbr_norm_var[b]+=d
            vn_std[a]+=d; vn_std[b]+=d
            dot=np.clip(np.abs(np.dot(normals[a],normals[b])),0,1)
            ang=float(np.arccos(dot)*180/np.pi)
            dihedral[a]+=ang; dihedral[b]+=ang; max_dih[a]=max(max_dih[a],ang); max_dih[b]=max(max_dih[b],ang)
            nbr_cnt[a]+=1; nbr_cnt[b]+=1
        mask=nbr_cnt>0; dihedral[mask]/=nbr_cnt[mask]

    x=compute_features(verts,faces,normals,centers,areas,span)
    ei=adj.T if len(adj)>0 else np.zeros((2,1),np.int64)
    edge_attr=np.zeros((len(adj),3),np.float32)
    if len(adj)>0:
        for ei_idx,(a,b) in enumerate(adj):
            ca,cb=centers[a],centers[b]; dot=np.clip(np.abs(np.dot(normals[a],normals[b])),0,1)
            edge_attr[ei_idx,0]=float(np.arccos(dot)); edge_attr[ei_idx,1]=np.linalg.norm(ca-cb)/max(span,1e-6)
            edge_attr[ei_idx,2]=float(np.dot(normals[a],cb-ca))

    # GAT
    model=TriangleGAT(in_dim=26)
    model.load_state_dict(torch.load(str(ROOT/"model.pt"),map_location='cpu',weights_only=False))
    model.eval()
    data=Data(x=torch.tensor(x),edge_index=torch.tensor(ei,dtype=torch.long),edge_attr=torch.tensor(edge_attr))
    with torch.no_grad(): pred=model(data).argmax(dim=1).numpy()

    # MLP edge + merge
    edge_models=[]
    for i in range(3):
        ckpt=torch.load(str(ROOT/f"edge_classifier_{i}.pt"),map_location='cpu',weights_only=False)
        m=EdgeMLP(); m.load_state_dict(ckpt['model']); m.eval()
        edge_models.append((m,ckpt['mean'],ckpt['std'],ckpt.get('threshold',0.5)))

    edge_feats=np.zeros((len(adj),4),np.float32)
    for ei,(a,b) in enumerate(adj):
        na,nb=normals[a],normals[b]; ca,cb=centers[a],centers[b]; dot=np.clip(np.dot(na,nb),-1,1)
        edge_feats[ei,0]=float(np.arccos(dot)*180/np.pi); edge_feats[ei,1]=min(areas[a],areas[b])/max(max(areas[a],areas[b]),1e-12)
        edge_feats[ei,2]=np.linalg.norm(ca-cb)/max(span,1e-6); edge_feats[ei,3]=float(np.dot(na,cb-ca))

    probs=[]
    for em,mean,std,thr in edge_models:
        fn=(edge_feats-mean)/std.clip(1e-6)
        with torch.no_grad(): probs.append(torch.sigmoid(em(torch.tensor(fn))).numpy())
    same_face=np.mean(probs,axis=0)>0.5

    nbrs=[[] for _ in range(len(faces))]
    for (a,b),s in zip(adj,same_face):
        if s: nbrs[a].append(b); nbrs[b].append(a)

    visited=np.zeros(len(faces),dtype=bool); regions=[]
    for seed in range(len(faces)):
        if visited[seed]: continue
        region=[seed]; visited[seed]=True; head=0
        while head<len(region):
            ti=region[head]; head+=1
            for nb in nbrs[ti]:
                if not visited[nb]: visited[nb]=True; region.append(nb)
        labels_in_region=pred[region]; area_weights=areas[region]
        weighted=np.bincount(labels_in_region,weights=area_weights,minlength=8)
        regions.append((region,int(np.argmax(weighted))))

    faces_out=[]
    for rt, rl in regions:
        tris=faces[rt]; name=LABEL_NAMES[rl]
        vset={}; vi=0; loc_v=[]; loc_t=[]
        for t in tris:
            lt=[]
            for v in t:
                if v not in vset: vset[v]=vi; loc_v.extend(verts[v].tolist()); vi+=1
                lt.append(vset[v])
            loc_t.extend(lt)
        cx=sum(loc_v[0::3])/vi; cy=sum(loc_v[1::3])/vi; cz=sum(loc_v[2::3])/vi
        faces_out.append({"vertices":loc_v,"triangles":loc_t,"type":name,"color":COLORS[name],"center":[cx,cy,cz]})

    return {"faces":faces_out,"center":centers.mean(0).tolist(),"span":span,"num_faces":len(faces_out)}


if __name__=="__main__":
    if len(sys.argv)<2: print(json.dumps({"error":"usage: python infer_standalone.py <stl_path>"})); sys.exit(1)
    print(json.dumps(predict(sys.argv[1])))
