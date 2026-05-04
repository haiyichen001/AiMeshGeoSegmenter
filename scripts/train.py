"""
GraphSAGE 面分类器 — 26 维输入, k-fold CV, JSON 训练日志
"""
import os, json as _json, random, time, collections
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import add_self_loops

ROOT = Path(r"D:\AiMeshGeoSegmenter")
GRAPH_DIR = ROOT / "data" / "graphs"
MODEL_DIR = ROOT / "models"
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_NAMES = ["plane", "cylinder", "sphere", "cone", "torus", "fillet", "chamfer", "freeform"]
NUM_CLASSES = len(LABEL_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM, NUM_LAYERS, DROPOUT = 128, 3, 0.3
BATCH_SIZE, LR, EPOCHS, K = 512, 0.001, 120, 5


def auto_tune_batch(model, graphs):
    """Find max batch size for GNN by probing VRAM."""
    candidates = [64, 128, 256, 512, 1024, 2048, 4096]
    best = 64
    for bs in candidates:
        if bs > len(graphs): break
        try:
            torch.cuda.empty_cache()
            loader = DataLoader(graphs[:bs], batch_size=bs, shuffle=True)
            batch = next(iter(loader)).to(DEVICE)
            model.train(); out = model(batch)
            l = F.nll_loss(out, batch.y); l.backward()
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            pct = used / total * 100
            best = bs
            del batch, out, l
            if pct > 80:
                print(f"  BS={bs}: {pct:.0f}% VRAM, sufficient")
                break
        except RuntimeError:
            torch.cuda.empty_cache()
            break
    return best


class FaceClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim)); self.norms.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim)); self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim//2, num_classes))

    def forward(self, data, drop_edge=0.0):
        x, ei = data.x, data.edge_index
        ei, _ = add_self_loops(ei, num_nodes=x.size(0))
        # DropEdge: randomly drop edges during training
        if drop_edge > 0 and self.training:
            mask = torch.rand(ei.size(1), device=ei.device) > drop_edge
            ei = ei[:, mask]
        x0 = self.res_proj(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_new = conv(x, ei); x_new = norm(x_new); x_new = F.relu(x_new); x_new = self.dropout(x_new)
            if i < len(self.convs) - 1:
                x = x_new + (x0 if x.shape == x_new.shape else x_new * 0)
            else:
                x = x_new
        return F.log_softmax(self.mlp(x), dim=-1)


def load_graph(fp):
    d = np.load(fp)
    return Data(x=torch.tensor(d["x"], dtype=torch.float32),
                edge_index=torch.tensor(d["edge_index"], dtype=torch.long),
                y=torch.tensor(d["y"], dtype=torch.long), num_nodes=int(d["num_nodes"]))


def evaluate(model, loader, class_weights):
    model.eval(); correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE); out = model(batch)
            loss = F.nll_loss(out, batch.y, weight=class_weights)
            loss_sum += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item(); total += batch.y.size(0)
    return correct / max(total, 1), loss_sum / len(loader.dataset)


if __name__ == "__main__":
    files = sorted(GRAPH_DIR.glob("*.npz"))
    graphs = [load_graph(f) for f in files]
    print(f"Loaded {len(graphs)} graphs")
    random.seed(42); random.shuffle(graphs)

    fold_size = len(graphs) // K
    folds = [graphs[i*fold_size:(i+1)*fold_size] for i in range(K)]
    for i in range(len(graphs) - K*fold_size): folds[i].append(graphs[K*fold_size + i])

    in_dim = graphs[0].x.shape[1]
    print(f"Input dim: {in_dim}, Device: {DEVICE}")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB, TF32 enabled")
    fold_accs = []; fold_histories = []

    for fold_idx in range(K):
        print(f"\n{'='*50}\nFOLD {fold_idx+1}/{K}\n{'='*50}")
        val_graphs = folds[fold_idx]
        train_graphs = [g for i in range(K) if i != fold_idx for g in folds[i]]
        print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE*2, shuffle=False)

        all_labels = []; _ = [all_labels.extend(g.y.numpy().tolist()) for g in train_graphs]
        lc = collections.Counter(all_labels); t = sum(lc.values())
        class_weights = torch.zeros(NUM_CLASSES)
        for i in range(NUM_CLASSES): class_weights[i] = (t / max(lc.get(i, 1), 1)) ** 0.5  # sqrt for softer rebalance
        class_weights = class_weights.to(DEVICE)

        model = FaceClassifier(in_dim, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, DROPOUT).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        if fold_idx == 0:
            BATCH_SIZE = auto_tune_batch(model, train_graphs)
            print(f"  Using BS={BATCH_SIZE}")
            train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE*2, shuffle=False)

        best_val_acc = 0; history = {'train_loss': [], 'val_acc': []}
        for epoch in range(1, EPOCHS+1):
            model.train(); tl, nb = 0, 0
            for batch in train_loader:
                batch = batch.to(DEVICE)
                # Rotation augmentation
                axis = torch.randn(3, device=DEVICE); axis = axis / axis.norm()
                angle = torch.rand(1, device=DEVICE).item() * 2 * 3.14159
                Kmat = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=DEVICE, dtype=torch.float32)
                R = torch.eye(3, device=DEVICE) + torch.sin(torch.tensor(angle)) * Kmat + (1 - torch.cos(torch.tensor(angle))) * (Kmat @ Kmat)
                batch.x[:, 1:4] = batch.x[:, 1:4] @ R.T; batch.x[:, 5:8] = batch.x[:, 5:8] @ R.T
                # Feature noise regularization
                noise = torch.randn_like(batch.x) * 0.01
                batch.x = batch.x + noise
                opt.zero_grad(); out = model(batch, drop_edge=0.2)
                loss = F.cross_entropy(out, batch.y, weight=class_weights, label_smoothing=0.1)
                loss.backward(); opt.step(); tl += loss.item() * batch.num_graphs; nb += 1
            sched.step()
            val_acc, _ = evaluate(model, val_loader, class_weights)
            history['train_loss'].append(tl / len(train_graphs)); history['val_acc'].append(val_acc)
            if val_acc > best_val_acc: best_val_acc = val_acc
            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | train_loss={history['train_loss'][-1]:.3f} val_acc={val_acc:.3f}")

        fold_accs.append(best_val_acc); fold_histories.append({'fold': fold_idx+1, 'loss': history['train_loss'], 'acc': history['val_acc']})
        print(f"  Fold {fold_idx+1} best val acc: {best_val_acc:.4f}")

    print(f"\n{'='*50}\nK-FOLD RESULTS (k={K})\n{'='*50}")
    print(f"Folds: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean: {np.mean(fold_accs):.4f}  Std: {np.std(fold_accs):.4f}")

    gnn_log = {"model": "GNN Face Classifier", "k": K, "epochs": EPOCHS,
               "fold_accuracies": [float(a) for a in fold_accs],
               "mean": float(np.mean(fold_accs)), "std": float(np.std(fold_accs)),
               "fold_histories": fold_histories}

    print(f"\nTraining final model on all {len(graphs)} graphs...")
    all_loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
    final_model = FaceClassifier(in_dim, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    all_labels = []; _ = [all_labels.extend(g.y.numpy().tolist()) for g in graphs]
    lc = collections.Counter(all_labels); t = sum(lc.values())
    cw = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES): cw[i] = t / max(lc.get(i, 1), 1)
    cw = cw.to(DEVICE)
    final_hist = {'loss': []}
    for epoch in range(1, EPOCHS+1):
        final_model.train(); tl, nb = 0, 0
        for batch in all_loader:
            batch = batch.to(DEVICE)
            axis = torch.randn(3, device=DEVICE); axis = axis / axis.norm()
            angle = torch.rand(1, device=DEVICE).item() * 2 * 3.14159
            Kmat = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=DEVICE, dtype=torch.float32)
            R = torch.eye(3, device=DEVICE) + torch.sin(torch.tensor(angle)) * Kmat + (1 - torch.cos(torch.tensor(angle))) * (Kmat @ Kmat)
            batch.x[:, 1:4] = batch.x[:, 1:4] @ R.T; batch.x[:, 5:8] = batch.x[:, 5:8] @ R.T
            noise = torch.randn_like(batch.x) * 0.01
            batch.x = batch.x + noise
            opt.zero_grad(); out = final_model(batch, drop_edge=0.2)
            loss = F.cross_entropy(out, batch.y, weight=cw, label_smoothing=0.1); loss.backward(); opt.step()
            tl += loss.item() * batch.num_graphs; nb += 1
        sched.step(); final_hist['loss'].append(tl / len(graphs))

    torch.save(final_model.state_dict(), MODEL_DIR / "face_classifier.pt")
    gnn_log["final"] = final_hist
    with open(MODEL_DIR / "gnn_train_log.json", "w") as f: _json.dump(gnn_log, f, indent=2)
    print(f"Final model saved: {os.path.getsize(MODEL_DIR / 'face_classifier.pt')/1024:.0f} KB")

