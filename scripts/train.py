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
    def __init__(self, in_dim=12, hidden=192, heads=4, n_classes=6, n_layers=3, dropout=0.3):
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
    x = np.stack([normals[:,0],normals[:,1],normals[:,2], rel_ctr[:,0],rel_ctr[:,1],rel_ctr[:,2],
                  area_log,center_dist,nbr_norm_var, dihedral, max_dih, shape], axis=1)
    ei = adj.T if len(adj)>0 else np.zeros((2,1),dtype=np.int64)
    return Data(x=torch.tensor(x), edge_index=torch.tensor(ei, dtype=torch.long),
                y=torch.tensor(labels), num_nodes=len(areas))


if __name__ == "__main__":
    files = sorted(DATA_DIR.glob("*.npz"))
    random.seed(42); random.shuffle(files)
    files = files[:2000]
    print(f"Loading {len(files)} parts..."); t0=time.time()
    graphs = [load_part(f) for f in files]
    print(f"Loaded in {time.time()-t0:.0f}s")

    if K == 1:
        random.seed(42); random.shuffle(files)
        n_tr = int(len(files)*0.8)
        splits = [(list(range(n_tr)), list(range(n_tr, len(files))))]
    else:
        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        splits = list(kf.split(files))
    fold_accs = []; fold_histories = []; BS = 8

    for fold_idx, (tidx, vidx) in enumerate(splits):
        print(f"\n{'='*50}\nFOLD {fold_idx+1}/{K}\n{'='*50}")
        tg = [graphs[i] for i in tidx]; vg = [graphs[i] for i in vidx]
        print(f"Train: {len(tg)}, Val: {len(vg)}")

        model = TriangleGAT(in_dim=graphs[0].x.shape[1], hidden=HIDDEN, heads=HEADS,
                            n_classes=NC, n_layers=LAYERS, dropout=DROPOUT).to(DEVICE)
        if fold_idx == 0:
            BS = auto_batch(model, tg); n_params = sum(p.numel() for p in model.parameters())
            print(f"  Params: {n_params:,}")

        tl = DataLoader(tg, batch_size=BS, shuffle=True); vl = DataLoader(vg, batch_size=BS*2, shuffle=False)
        all_l = np.concatenate([g.y.numpy() for g in tg]); tc = sum(all_l); cw = torch.zeros(NC)
        for i in range(NC): cw[i] = (tc/max(collections.Counter(all_l).get(i,1),1))**0.5
        cw = cw.to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        best = 0; patience = 0; hist = {'loss':[], 'val_loss':[], 'acc':[]}
        for epoch in range(1, EPOCHS+1):
            model.train(); tls, nb = 0, 0; check_vram()
            for batch in tl:
                batch = batch.to(DEVICE); opt.zero_grad()
                loss = focal_loss(model(batch), batch.y, alpha=cw, gamma=2.0)
                loss.backward(); opt.step(); tls += loss.item()*batch.num_graphs; nb += 1
            hist['loss'].append(tls/len(tg))
            model.eval(); correct, total, vls = 0, 0, 0
            with torch.no_grad():
                for batch in vl:
                    batch = batch.to(DEVICE)
                    logits = model(batch)
                    vls += focal_loss(logits, batch.y, alpha=cw).item() * batch.num_graphs
                    pred = logits.argmax(dim=1)
                    correct += (pred==batch.y).sum().item(); total += batch.y.size(0)
            hist['val_loss'].append(vls/len(vg))
            acc = correct/max(total,1); hist['acc'].append(acc)
            if acc > best: best = acc; patience = 0
            else: patience += 1
            if epoch%10==0 or epoch==1: print(f"  Epoch {epoch:3d} | loss={hist['loss'][-1]:.3f} val_loss={hist['val_loss'][-1]:.3f} acc={acc:.3f} | patience={patience}/50")
            if patience >= 50: print(f"  Early stopping at epoch {epoch}"); break
        fold_accs.append(best); fold_histories.append({'fold':fold_idx+1,'loss':hist['loss'],'val_loss':hist['val_loss'],'acc':hist['acc']})
        print(f"  Best: {best:.4f}")

    print(f"\n{'='*50}\nK-FOLD (k={K})\n{'='*50}")
    print(f"Folds: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean: {np.mean(fold_accs):.4f}  Std: {np.std(fold_accs):.4f}")

    print(f"\nTraining final model on all {len(graphs)} parts...")
    fm = TriangleGAT(in_dim=graphs[0].x.shape[1], hidden=HIDDEN, heads=HEADS,
                     n_classes=NC, n_layers=LAYERS, dropout=DROPOUT).to(DEVICE)
    al = DataLoader(graphs, batch_size=BS, shuffle=True)
    all_l = np.concatenate([g.y.numpy() for g in graphs]); tc = sum(all_l); cw = torch.zeros(NC)
    for i in range(NC): cw[i] = (tc/max(collections.Counter(all_l).get(i,1),1))**0.5; cw = cw.to(DEVICE)
    opt = torch.optim.AdamW(fm.parameters(), lr=0.001, weight_decay=1e-4)
    final_hist = {'loss':[]}
    for epoch in range(1, EPOCHS+1):
        fm.train(); tls, nb = 0, 0
        for batch in al:
            batch = batch.to(DEVICE); opt.zero_grad()
            loss = focal_loss(fm(batch), batch.y, alpha=cw, gamma=2.0)
            loss.backward(); opt.step(); tls += loss.item()*batch.num_graphs; nb += 1
        final_hist['loss'].append(tls/len(graphs))

    torch.save(fm.state_dict(), MODEL_DIR/"model.pt")
    log = {"model":"GAT","k":K,"epochs":EPOCHS,"params":n_params,"batch_size":BS,
           "fold_accuracies":[float(a) for a in fold_accs],
           "mean":float(np.mean(fold_accs)),"std":float(np.std(fold_accs)),
           "fold_histories":fold_histories,"final":final_hist}
    with open(MODEL_DIR/"train_log.json","w") as f: json.dump(log, f, indent=2)
    print(f"Model: {os.path.getsize(MODEL_DIR/'model.pt')/1024:.0f} KB")
