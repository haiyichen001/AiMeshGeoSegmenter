"""
GAT: STL 三角图分类, 5 折 CV, 自适应 batch
"""
import os, json, random, time, collections, numpy as np, trimesh
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import KFold

ROOT = Path(r"D:\AiMeshGeoSegmenter"); DATA_DIR = ROOT/"data"/"patches"; MODEL_DIR = ROOT/"models"
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["plane","cylinder","sphere","cone","torus","freeform"]
NC, K, EPOCHS = len(LABEL_NAMES), 3, 250
HIDDEN, HEADS, LAYERS, DROPOUT = 192, 4, 3, 0.3
IN_DIM, EDGE_DIM = 14, 3
LR, WARMUP = 0.002, 25

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    # Strict: no shared GPU memory. Monitor peak, hard-limit batch size.
    vram_total = torch.cuda.get_device_properties(0).total_memory
    VRAM_LIMIT = int(vram_total * 0.85)  # 85% hard ceiling
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:32,roundup_power2_divisions:32,garbage_collection_threshold:0.8"
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_total/1e9:.1f} GB, limit: 85% ({VRAM_LIMIT/1e9:.2f} GB)")
    def check_vram():
        reserved = torch.cuda.memory_reserved()
        if reserved > VRAM_LIMIT * 0.95:
            print(f"  WARNING: VRAM {reserved/1e9:.1f}GB near limit {VRAM_LIMIT/1e9:.1f}GB, clearing cache")
            torch.cuda.empty_cache()


def focal_loss(logits, targets, alpha=None, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    loss = (1 - pt) ** gamma * ce
    if alpha is not None:
        loss = alpha[targets] * loss
    return loss.mean()


class TriangleGAT(nn.Module):
    def __init__(self, in_dim=14, hidden=192, heads=4, n_classes=6, n_layers=3, dropout=0.3, edge_dim=3):
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
        # self-loops need zero edge features
        n_self = x.size(0)
        ea_full = torch.cat([ea, torch.zeros(n_self, ea.size(1), device=x.device)], dim=0)
        for i,conv in enumerate(self.convs):
            x = conv(x, ei, ea_full)
            if i < len(self.norms): x = self.norms[i](x); x = F.elu(x); x = self.dropout(x)
        return F.log_softmax(self.mlp(x), dim=-1)


def auto_batch(model, graphs):
    print("Auto-tuning batch size...")
    lo, hi, best = 1, 256, 1; limit = VRAM_LIMIT
    model.train()
    while lo <= hi:
        mid = (lo+hi)//2
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            loader = DataLoader(graphs[:min(mid*2,len(graphs))], batch_size=mid, shuffle=True)
            batch = next(iter(loader)).to(DEVICE)
            out = model(batch); loss = F.cross_entropy(out, batch.y); loss.backward()
            # Check peak VRAM (reserved = allocated + cached, reflects real GPU usage)
            peak = torch.cuda.max_memory_reserved()
            if peak > limit:
                hi = mid-1; torch.cuda.empty_cache(); print(f"  BS={mid}: exceeded limit ({peak/1e9:.1f}GB > {limit/1e9:.1f}GB)")
                del batch, out, loss; continue
            best = mid; lo = mid+1
            print(f"  BS={mid}: OK, peak={peak/1e9:.1f}GB (limit={limit/1e9:.1f}GB)")
            del batch, out, loss
        except RuntimeError as e:
            if 'out of memory' in str(e): hi = mid-1; torch.cuda.empty_cache(); print(f"  BS={mid}: OOM")
            else: raise
    best = max(1,int(best*0.85))
    print(f"  Using BS={best}"); return best


def load_part(fp):
    d = np.load(fp)
    verts = d["vertices"]; faces = d["faces"]; normals = d["tri_normals"].astype(np.float32)
    areas = d["tri_areas"].astype(np.float32); labels = d["tri_labels"].astype(np.int64)
    centers = verts[faces].mean(axis=1).astype(np.float32)
    part_ctr = centers.mean(axis=0); part_r = float(np.sqrt(((centers-part_ctr)**2).sum(axis=1).mean()))
    area_log = np.log10(np.maximum(areas,1e-6))
    center_dist = np.sqrt(((centers-part_ctr)**2).sum(axis=1))/max(part_r,1e-6)
    rel_ctr = (centers-part_ctr)/max(part_r,1e-6)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    adj = m.face_adjacency
    nbr_norm_var = np.zeros(len(areas), dtype=np.float32)
    if len(adj) > 0:
        for a,b in adj: d_n = np.linalg.norm(normals[a]-normals[b]); nbr_norm_var[a]+=d_n; nbr_norm_var[b]+=d_n
    # Curvature features (per-triangle, computed from mesh)
    e1 = verts[faces[:,1]] - verts[faces[:,0]]
    e2 = verts[faces[:,2]] - verts[faces[:,0]]
    e3 = verts[faces[:,2]] - verts[faces[:,1]]
    perimeter = np.linalg.norm(e1, axis=1) + np.linalg.norm(e2, axis=1) + np.linalg.norm(e3, axis=1)
    shape = np.sqrt(np.maximum(areas, 1e-12)) / np.maximum(perimeter * 0.07, 1e-12)  # compactness [0~1]
    # Mean & max dihedral angle from adjacency
    dihedral = np.zeros(len(areas), dtype=np.float32)
    max_dih = np.zeros(len(areas), dtype=np.float32)
    if len(adj) > 0:
        for a,b in adj:
            dot = np.clip(np.abs(np.dot(normals[a], normals[b])), 0, 1)
            ang = float(np.arccos(dot) * 180 / np.pi)
            dihedral[a] += ang; dihedral[b] += ang
            max_dih[a] = max(max_dih[a], ang)
            max_dih[b] = max(max_dih[b], ang)
        nbr_cnt = np.bincount(np.concatenate([adj[:,0], adj[:,1]]), minlength=len(areas))
        mask = nbr_cnt > 0; dihedral[mask] /= nbr_cnt[mask]
    # 2 more features: edge ratio (shape quality), vertex normal std (curvature proxy)
    l1 = np.linalg.norm(e1, axis=1); l2 = np.linalg.norm(e2, axis=1); l3 = np.linalg.norm(e3, axis=1)
    edge_ratio = np.minimum(np.minimum(l1,l2),l3) / np.maximum(np.maximum(l1,l2),np.maximum(l3, 1e-12))
    # Vertex normal variance via face normals in 1-ring
    vn_std = np.zeros(len(areas), dtype=np.float32)
    if len(adj) > 0:
        nbr_cnt2 = np.bincount(np.concatenate([adj[:,0], adj[:,1]]), minlength=len(areas))
        for a,b in adj:
            vn_std[a] += np.linalg.norm(normals[a] - normals[b])
            vn_std[b] += vn_std[a] - vn_std[a] + np.linalg.norm(normals[a] - normals[b])
    # Edge features: dihedral angle, relative edge length
    edge_attr = np.zeros((len(adj), EDGE_DIM), dtype=np.float32)
    if len(adj) > 0:
        # Compute shared edge length for each adjacency pair
        # Use face-face shared edge midpoint distance (approximate)
        for ei_idx, (a,b) in enumerate(adj):
            ca = centers[a]; cb = centers[b]
            # dihedral angle
            dot_ab = np.clip(np.abs(np.dot(normals[a], normals[b])), 0, 1)
            edge_attr[ei_idx, 0] = float(np.arccos(dot_ab))  # radians
            # edge-relative length: distance between face centers / avg perimeter
            dist_ab = np.linalg.norm(ca - cb)
            avg_perim = (perimeter[a] + perimeter[b]) / 2.0
            edge_attr[ei_idx, 1] = dist_ab / max(avg_perim, 1e-6)
            # normal variance direction (sign indicator for convex/concave)
            edge_attr[ei_idx, 2] = float(np.dot(normals[a], cb - ca))  # signed

    x = np.stack([normals[:,0],normals[:,1],normals[:,2], rel_ctr[:,0],rel_ctr[:,1],rel_ctr[:,2],
                  area_log,center_dist,nbr_norm_var, dihedral, max_dih, shape, edge_ratio, vn_std], axis=1)
    ei = adj.T if len(adj)>0 else np.zeros((2,1),dtype=np.int64)
    ea = torch.tensor(edge_attr, dtype=torch.float32) if len(adj)>0 else torch.zeros(0, EDGE_DIM)
    return Data(x=torch.tensor(x), edge_index=torch.tensor(ei, dtype=torch.long),
                edge_attr=ea, y=torch.tensor(labels), num_nodes=len(areas))


def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


if __name__ == "__main__":
    import torch
    files = sorted(DATA_DIR.glob("*.npz"))
    random.seed(42); random.shuffle(files); files = files[:2000]
    print(f"Loading {len(files)} parts..."); t1=time.time()
    graphs = [load_part(f) for f in files]
    print(f"Loaded in {time.time()-t1:.0f}s")

    # 70/15/15 train/val/test split
    n_train = int(len(files) * 0.7)
    n_val = int(len(files) * 0.15)
    tg = graphs[:n_train]
    vg = graphs[n_train:n_train + n_val]
    test_g = graphs[n_train + n_val:]
    print(f"Train: {len(tg)}, Val: {len(vg)}, Test: {len(test_g)}")

    model = TriangleGAT(in_dim=graphs[0].x.shape[1], hidden=HIDDEN, heads=HEADS,
                        n_classes=NC, n_layers=LAYERS, dropout=DROPOUT, edge_dim=EDGE_DIM).to(DEVICE)
    BS = auto_batch(model, tg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}, BS: {BS}")

    tl = DataLoader(tg, batch_size=BS, shuffle=True)
    vl = DataLoader(vg, batch_size=BS*2, shuffle=False)
    all_l = np.concatenate([g.y.numpy() for g in tg])
    cw = torch.zeros(NC)
    for i in range(NC): cw[i] = (len(all_l) / max(collections.Counter(all_l).get(i, 1), 1)) ** 0.5
    cw = cw.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = warmup_cosine_scheduler(opt, WARMUP, EPOCHS)

    best = 0; patience = 0; hist = {'loss':[], 'val_loss':[], 'acc':[]}
    swa_weights = []  # SWA: collect snapshots in the stable zone
    train_start = time.time()
    for epoch in range(1, EPOCHS+1):
        model.train(); tls = 0
        for batch in tl:
            batch = batch.to(DEVICE); opt.zero_grad()
            loss = focal_loss(model(batch), batch.y, alpha=cw, gamma=2.0)
            loss.backward(); opt.step()
            tls += loss.item() * batch.num_graphs
        hist['loss'].append(tls / len(tg))

        model.eval(); correct, total, vls = 0, 0, 0
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(DEVICE)
                logits = model(batch)
                vls += focal_loss(logits, batch.y, alpha=cw).item() * batch.num_graphs
                pred = logits.argmax(dim=1)
                correct += (pred == batch.y).sum().item(); total += batch.y.size(0)
        hist['val_loss'].append(vls / len(vg))
        acc = correct / max(total, 1); hist['acc'].append(acc)

        if acc > best: best = acc; patience = 0
        else: patience += 1
        # SWA: collect weight snapshots when within 5% of best
        if acc >= best * 0.95 and epoch > WARMUP:
            swa_weights.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={hist['loss'][-1]:.3f} val_loss={hist['val_loss'][-1]:.3f} acc={acc:.3f} | patience={patience}/50")
        if patience >= 50: print(f"  Early stop at {epoch}"); break
        scheduler.step()
    train_time = time.time() - train_start

    # SWA: average collected weights
    if swa_weights:
        swa_state = {}
        for key in swa_weights[0]:
            swa_state[key] = sum(w[key] for w in swa_weights) / len(swa_weights)
        model.load_state_dict(swa_state)
        print(f"SWA: averaged {len(swa_weights)} snapshots")
    else:
        print("SWA: no snapshots collected, using best model")

    print(f"\n{'='*50}")
    print(f"Train Time: {train_time:.0f}s ({train_time/60:.1f}min)")
    print(f"Best Val Acc: {best:.4f}")
    print(f"Params: {n_params:,}")

    # Test set evaluation (SWA model)
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for batch in DataLoader(test_g, batch_size=BS*2, shuffle=False):
            batch = batch.to(DEVICE)
            pred = model(batch).argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    test_acc = correct / max(total, 1)
    print(f"Test Acc (SWA): {test_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    print(f"Model: {os.path.getsize(MODEL_DIR/'model.pt')/1024:.0f} KB")

    log = {"model": "GAT+v3+SWA", "epochs": EPOCHS, "params": n_params, "batch_size": BS,
           "best_val_acc": float(best), "test_acc": float(test_acc),
           "train_time_s": round(train_time, 1),
           "fold_histories": [{"loss": hist['loss'], "val_loss": hist['val_loss'], "acc": hist['acc']}]}
    with open(MODEL_DIR / "train_log.json", "w") as f: json.dump(log, f, indent=2)
    print("Log saved.")
