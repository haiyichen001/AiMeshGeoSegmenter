"""
GAT 三角面分类: STL 三角图上直接训练, 5 折 CV
"""
import os, json, random, time, collections, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import KFold

ROOT = Path(r"D:\AiMeshGeoSegmenter")
DATA_DIR = ROOT / "data" / "patches"
MODEL_DIR = ROOT / "models"
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["plane","cylinder","sphere","cone","torus","fillet","chamfer","freeform"]
NC = len(LABEL_NAMES)
K, EPOCHS, BS = 5, 50, 8
HIDDEN, HEADS, LAYERS, DROPOUT = 128, 4, 3, 0.3

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


class TriangleGAT(nn.Module):
    """GAT on STL triangle adjacency graph."""
    def __init__(self, in_dim=10, hidden=128, heads=4, n_classes=8, n_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden*heads]*n_layers
        for i in range(n_layers):
            in_c = dims[i]; out_c = hidden if i < n_layers-1 else hidden//2
            h = heads if i < n_layers-1 else 1
            self.convs.append(GATConv(in_c, out_c, heads=h, dropout=dropout))
            if i < n_layers-1: self.norms.append(nn.BatchNorm1d(out_c*h))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden//2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, data):
        x, ei = data.x, data.edge_index
        ei, _ = add_self_loops(ei, num_nodes=x.size(0))
        for i, conv in enumerate(self.convs):
            x = conv(x, ei)
            if i < len(self.norms): x = self.norms[i](x); x = F.elu(x); x = self.dropout(x)
        return F.log_softmax(self.mlp(x), dim=-1)


def load_part(fp):
    d = np.load(fp)
    # Features: normal(3) + center(3) + area_log(1) + center_dist(1) + edge_ratio(1) + curvature_proxy(1)
    verts = d["vertices"]; faces = d["faces"]
    normals = d["tri_normals"].astype(np.float32)
    areas = d["tri_areas"].astype(np.float32)
    labels = d["tri_labels"].astype(np.int64)

    # Triangle centers
    centers = verts[faces].mean(axis=1).astype(np.float32)
    part_ctr = centers.mean(axis=0)
    part_r = float(np.sqrt(((centers-part_ctr)**2).sum(axis=1).mean()))

    # Per-triangle features
    area_log = np.log10(np.maximum(areas, 1e-6))
    center_dist = np.sqrt(((centers - part_ctr)**2).sum(axis=1)) / max(part_r, 1e-6)
    rel_center = (centers - part_ctr) / max(part_r, 1e-6)

    # Curvature proxy: normal variance in 1-ring
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    adj = mesh.face_adjacency
    nbr_norm_var = np.zeros(len(areas), dtype=np.float32)
    if len(adj) > 0:
        import trimesh
        for a, b in adj:
            diff = np.linalg.norm(normals[a]-normals[b])
            nbr_norm_var[a] += diff; nbr_norm_var[b] += diff

    x = np.stack([normals[:,0], normals[:,1], normals[:,2],
                  rel_center[:,0], rel_center[:,1], rel_center[:,2],
                  area_log, center_dist, nbr_norm_var, np.zeros(len(areas), dtype=np.float32)], axis=1)

    # Edge index from mesh adjacency
    ei_raw = mesh.face_adjacency.T if len(mesh.face_adjacency) > 0 else np.zeros((2,1), dtype=np.int64)

    return Data(x=torch.tensor(x), edge_index=torch.tensor(ei_raw, dtype=torch.long),
                y=torch.tensor(labels), num_nodes=len(areas))


if __name__ == "__main__":
    files = sorted(DATA_DIR.glob("*.npz"))
    print(f"Loading {len(files)} parts...")
    t0 = time.time()
    graphs = [load_part(f) for f in files]
    print(f"Loaded in {time.time()-t0:.0f}s")

    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    fold_accs = []; fold_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(files)):
        print(f"\n{'='*50}\nFOLD {fold_idx+1}/{K}\n{'='*50}")
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

        tl = DataLoader(train_graphs, batch_size=BS, shuffle=True)
        vl = DataLoader(val_graphs, batch_size=BS*2, shuffle=False)

        # Class weights
        all_labs = np.concatenate([g.y.numpy() for g in train_graphs])
        lc = collections.Counter(all_labs); t = sum(lc.values())
        cw = torch.zeros(NC)
        for i in range(NC): cw[i] = (t/max(lc.get(i,1),1))**0.5
        cw = cw.to(DEVICE)

        model = TriangleGAT(in_dim=graphs[0].x.shape[1], hidden=HIDDEN, heads=HEADS,
                            n_classes=NC, n_layers=LAYERS, dropout=DROPOUT).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        best = 0; hist = {'loss':[], 'acc':[]}
        for epoch in range(1, EPOCHS+1):
            model.train(); tl_sum, nb = 0, 0
            for batch in tl:
                batch = batch.to(DEVICE); opt.zero_grad()
                loss = F.cross_entropy(model(batch), batch.y, weight=cw, label_smoothing=0.05)
                loss.backward(); opt.step()
                tl_sum += loss.item() * batch.num_graphs; nb += 1
            hist['loss'].append(tl_sum/len(train_graphs))

            model.eval(); correct, total = 0, 0
            with torch.no_grad():
                for batch in vl:
                    batch = batch.to(DEVICE)
                    pred = model(batch).argmax(dim=1)
                    correct += (pred == batch.y).sum().item(); total += batch.y.size(0)
            acc = correct/max(total,1); hist['acc'].append(acc)
            if acc > best: best = acc
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | loss={hist['loss'][-1]:.3f} acc={acc:.3f}")

        fold_accs.append(best); fold_histories.append({'fold':fold_idx+1,'loss':hist['loss'],'acc':hist['acc']})
        print(f"  Best: {best:.4f}")

    print(f"\n{'='*50}\nK-FOLD (k={K})\n{'='*50}")
    print(f"Folds: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean: {np.mean(fold_accs):.4f}  Std: {np.std(fold_accs):.4f}")

    # Final model
    print(f"\nTraining final model on all {len(graphs)} parts...")
    al = DataLoader(graphs, batch_size=BS, shuffle=True)
    all_labs = np.concatenate([g.y.numpy() for g in graphs])
    lc = collections.Counter(all_labs); t = sum(lc.values())
    cw = torch.zeros(NC)
    for i in range(NC): cw[i] = (t/max(lc.get(i,1),1))**0.5
    cw = cw.to(DEVICE)
    fm = TriangleGAT(in_dim=graphs[0].x.shape[1], hidden=HIDDEN, heads=HEADS,
                     n_classes=NC, n_layers=LAYERS, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(fm.parameters(), lr=0.001, weight_decay=1e-4)
    final_hist = {'loss':[]}
    for epoch in range(1, EPOCHS+1):
        fm.train(); tl_sum, nb = 0, 0
        for batch in al:
            batch = batch.to(DEVICE); opt.zero_grad()
            loss = F.cross_entropy(fm(batch), batch.y, weight=cw, label_smoothing=0.05)
            loss.backward(); opt.step()
            tl_sum += loss.item() * batch.num_graphs; nb += 1
        final_hist['loss'].append(tl_sum/len(graphs))

    torch.save(fm.state_dict(), MODEL_DIR / "model.pt")
    log = {"model":"GAT","k":K,"epochs":EPOCHS,
           "fold_accuracies":[float(a) for a in fold_accs],
           "mean":float(np.mean(fold_accs)),"std":float(np.std(fold_accs)),
           "fold_histories":fold_histories,"final":final_hist}
    with open(MODEL_DIR/"train_log.json","w") as f: json.dump(log, f, indent=2)
    print(f"Model: {os.path.getsize(MODEL_DIR/'model.pt')/1024:.0f} KB")
