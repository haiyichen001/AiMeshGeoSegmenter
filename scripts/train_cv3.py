"""20K ensemble training: 3 seeds, 3-fold CV, GPU-direct, JK+AMP+DropEdge+SWA"""
import os, json, random, time, collections, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import KFold

ROOT = Path(r"D:\AiMeshGeoSegmenter")
MODEL_DIR = ROOT / "models"
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["plane","cylinder","sphere","cone","torus","freeform"]
NC, K, EPOCHS = len(LABEL_NAMES), 3, 250
HIDDEN, HEADS, LAYERS, DROPOUT = 192, 4, 3, 0.3
LR, WARMUP = 0.002, 25

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    scaler = torch.amp.GradScaler('cuda')
    vram_total = torch.cuda.get_device_properties(0).total_memory
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_total/1e9:.1f} GB")

def focal_loss(logits, targets, alpha=None, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    loss = (1 - pt) ** gamma * ce
    if alpha is not None: loss = alpha[targets] * loss
    return loss.mean()

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
        self.mlp = nn.Sequential(nn.Linear(jk_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(64, n_classes))
        self.drop_edge_prob = 0.15
    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        n_self = x.size(0)
        all_x = [x]
        for i, conv in enumerate(self.convs):
            if self.training and self.drop_edge_prob > 0:
                n_edges = ei.size(1)
                mask = torch.rand(n_edges, device=ei.device) > self.drop_edge_prob
                ei_drop = ei[:, mask]
                ei_drop, _ = add_self_loops(ei_drop, num_nodes=n_self)
                ea_drop = torch.cat([ea[mask], torch.zeros(n_self, ea.size(1), device=x.device)], dim=0)
                x = conv(x, ei_drop, ea_drop)
            else:
                ei_full, _ = add_self_loops(ei, num_nodes=n_self)
                ea_all = torch.cat([ea, torch.zeros(n_self, ea.size(1), device=x.device)], dim=0)
                x = conv(x, ei_full, ea_all)
            if i < len(self.norms): x = self.norms[i](x); x = F.elu(x); x = self.dropout(x)
            all_x.append(x)
        x_cat = torch.cat(all_x, dim=-1)
        return F.log_softmax(self.mlp(x_cat), dim=-1)

def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs: return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def rotate_normals_2d(nx, ny, angle):
    """Rotate normal vectors in XY plane."""
    c, s = np.cos(angle), np.sin(angle)
    rnx = c * nx - s * ny
    rny = s * nx + c * ny
    return rnx, rny

def augment_batch(graphs, noise_std=0.005):
    """Online augmentation: random Z-rotation + normal noise."""
    angle = random.uniform(0, 2 * np.pi)
    augmented = []
    for g in graphs:
        gc = g.clone()
        # Normal noise
        gc.x[:, :3] += torch.randn_like(gc.x[:, :3]) * noise_std
        # Renormalize normals
        nrm = gc.x[:, :3].norm(dim=1, keepdim=True).clamp(1e-8)
        gc.x[:, :3] = gc.x[:, :3] / nrm
        augmented.append(gc)
    return Batch.from_data_list(augmented)

def train_one_fold(model, tg, vg, cw, BS, fold_name, warmup, total_ep):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = warmup_cosine_scheduler(opt, warmup, total_ep)
    n_train, n_val = len(tg), len(vg)
    best = 0; patience = 0; hist = {'loss':[], 'val_loss':[], 'acc':[]}
    swa_weights = []
    for epoch in range(1, total_ep+1):
        model.train(); tls = 0
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train, BS):
            idx = perm[i:i+BS].tolist()
            batch = Batch.from_data_list([tg[j] for j in idx])
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = focal_loss(model(batch), batch.y, alpha=cw, gamma=2.0)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tls += loss.item() * len(idx)
        hist['loss'].append(tls / n_train)

        model.eval(); correct, total, vls = 0, 0, 0
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for i in range(0, n_val, BS*2):
                idx = list(range(i, min(i+BS*2, n_val)))
                batch = Batch.from_data_list([vg[j] for j in idx])
                logits = model(batch)
                vls += focal_loss(logits, batch.y, alpha=cw).item() * len(idx)
                pred = logits.argmax(dim=1)
                correct += (pred == batch.y).sum().item(); total += batch.y.size(0)
        hist['val_loss'].append(vls / n_val)
        acc = correct / max(total, 1); hist['acc'].append(acc)

        if acc > best: best = acc; patience = 0
        else: patience += 1
        if acc >= best * 0.95 and epoch > warmup:
            swa_weights.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 20 == 0 or epoch == 1:
            print(f"    {fold_name} E{epoch:3d} loss={hist['loss'][-1]:.3f} vloss={hist['val_loss'][-1]:.3f} acc={acc:.3f} p={patience}/50")
        if patience >= 50: print(f"    Early stop {epoch}"); break
        scheduler.step()

    if swa_weights:
        swa_state = {k: sum(w[k] for w in swa_weights) / len(swa_weights) for k in swa_weights[0]}
        model.load_state_dict(swa_state)
    return best, hist

if __name__ == "__main__":
    import torch, sys
    CACHE = ROOT / "data" / "graphs_20k.pt"
    log_f = open(MODEL_DIR / "train.log", "w", encoding="utf-8")
    class TeeIO:
        def __init__(self, f1, f2): self.f1 = f1; self.f2 = f2
        def write(self, s): self.f1.write(s); self.f1.flush(); self.f2.write(s); self.f2.flush()
        def flush(self): self.f1.flush(); self.f2.flush()
    sys.stdout = TeeIO(log_f, sys.stderr)

    print(f"Loading {CACHE.name}..."); t0 = time.time()
    graphs = torch.load(CACHE, map_location='cpu', weights_only=False)
    random.seed(42); random.shuffle(graphs)
    # Prune to 26-dim
    feat_idx = list(range(26))
    for g in graphs: g.x = g.x[:, feat_idx].contiguous()
    print(f"Loaded {len(graphs)} graphs ({graphs[0].x.shape[1]}-dim) in {time.time()-t0:.0f}s")

    # 70/15/15 split
    n_total = len(graphs)
    n_train = int(n_total * 0.7); n_val = int(n_total * 0.15)
    train_g = graphs[:n_train]
    val_g = graphs[n_train:n_train + n_val]
    test_g = graphs[n_train + n_val:]
    print(f"Split: train={len(train_g)} val={len(val_g)} test={len(test_g)}")

    # Move to GPU
    print("Moving to GPU...")
    train_g = [g.to(DEVICE) for g in train_g]
    val_g = [g.to(DEVICE) for g in val_g]
    test_g = [g.to(DEVICE) for g in test_g]
    print("All on GPU.")

    # Class weights
    all_l = np.concatenate([g.y.cpu().numpy() for g in train_g])
    cw = torch.zeros(NC)
    for i in range(NC): cw[i] = (len(all_l) / max(collections.Counter(all_l).get(i,1), 1)) ** 0.5
    cw = cw.to(DEVICE)

    # Auto-batch
    model = TriangleGAT(in_dim=26).to(DEVICE)
    model.train()
    best_bs = 128
    for bs in [128, 192, 224, 240, 248, 252, 254, 255, 256]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            batch = Batch.from_data_list([train_g[j] for j in range(min(bs*2, len(train_g)))])
            out = model(batch); loss = F.cross_entropy(out, batch.y); loss.backward()
            peak = torch.cuda.max_memory_reserved()
            if peak > vram_total * 0.80:
                print(f"  BS={bs}: exceeded limit ({peak/1e9:.1f}GB)"); break
            best_bs = bs
            print(f"  BS={bs}: OK, peak={peak/1e9:.1f}GB")
        except: break
    BS = max(1, int(best_bs * 0.85))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BS={BS}, Params={n_params:,}")
    del model; torch.cuda.empty_cache()

    # 3-fold CV (single model, no ensemble)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_histories = []

    for fold_idx, (t_idx, v_idx) in enumerate(kf.split(train_g)):
        print(f"\n{'='*50}\nFOLD {fold_idx+1}/3\n{'='*50}")
        tg = [train_g[i] for i in t_idx]; vg = [train_g[i] for i in v_idx]
        print(f"Train: {len(tg)}, Val: {len(vg)}")
        model = TriangleGAT(in_dim=26).to(DEVICE)
        best, hist = train_one_fold(model, tg, vg, cw, BS, f"F{fold_idx+1}", WARMUP, EPOCHS)
        fold_histories.append({'fold': fold_idx+1, 'loss': hist['loss'], 'val_loss': hist['val_loss'], 'acc': hist['acc'], 'best_val': float(best)})
        print(f"  Best: {best:.4f}")
        torch.save(model.state_dict(), MODEL_DIR / f"cv_fold{fold_idx+1}.pt")
        del model; torch.cuda.empty_cache()

    # Test evaluation on test set with best fold model
    print(f"\n{'='*50}\nTEST EVALUATION\n{'='*50}")
    best_fold = max(fold_histories, key=lambda h: h['best_val'])
    best_idx = best_fold['fold'] - 1
    model = TriangleGAT(in_dim=26).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / f"cv_fold{best_idx+1}.pt", map_location=DEVICE))
    model.eval()
    correct, total = 0, 0
    n_test = len(test_g)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(0, n_test, BS*2):
            idx = list(range(i, min(i+BS*2, n_test)))
            batch = Batch.from_data_list([test_g[j] for j in idx])
            pred = model(batch).argmax(dim=1)
            correct += (pred == batch.y).sum().item(); total += batch.y.size(0)
    test_acc = correct / max(total, 1)
    print(f"Test Acc: {test_acc:.4f}")
    del model; torch.cuda.empty_cache()

    # Save log
    mean_val = np.mean([h['best_val'] for h in fold_histories])
    train_time = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"Folds: {[round(h['best_val'],4) for h in fold_histories]}")
    print(f"Mean Val: {mean_val:.4f}, Test: {test_acc:.4f}, Time: {train_time/60:.0f}min")

    log = {"model": "GAT+JK+AMP 3-fold CV", "k": 3, "epochs": EPOCHS, "params": n_params, "batch_size": BS,
           "train_time_s": round(train_time, 0), "fold_histories": fold_histories,
           "best_val_acc": float(max(h['best_val'] for h in fold_histories)),
           "test_acc": float(test_acc), "mean_val": float(mean_val)}
    with open(MODEL_DIR / "train_log.json", "w") as f: json.dump(log, f, indent=2)
    print("Log saved.")
