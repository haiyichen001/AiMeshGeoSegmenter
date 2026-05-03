"""
GraphSAGE 面类型分类 — 训练脚本

模型: 3层 GraphSAGE + MLP classifier
输入: 图数据集 (data/graphs/*.npz)
输出: 训练好的模型 weights
"""
import os, json, random, time, collections
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops

ROOT = Path(r"D:\AiMeshGeoSegmenter")
GRAPH_DIR = ROOT / "data" / "graphs"
MODEL_DIR = ROOT / "models"
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_NAMES = ["plane", "cylinder", "sphere", "cone", "torus", "fillet", "chamfer", "freeform"]
NUM_CLASSES = len(LABEL_NAMES)
DEVICE = torch.device("cpu")

# ---- Hyperparams ----
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 80
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


class FaceClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.mlp(x)
        return F.log_softmax(x, dim=-1)


def load_graph(filepath):
    d = np.load(filepath)
    x = torch.tensor(d["x"], dtype=torch.float32)
    edge_index = torch.tensor(d["edge_index"], dtype=torch.long)
    y = torch.tensor(d["y"], dtype=torch.long)
    n = int(d["num_nodes"])
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=n)


def main():
    # Load all graphs
    files = sorted(GRAPH_DIR.glob("*.npz"))
    print(f"Loading {len(files)} graphs...")
    t0 = time.time()
    graphs = [load_graph(f) for f in files]
    print(f"Loaded in {time.time()-t0:.1f}s")

    # K-fold cross-validation
    K = 5
    random.seed(42)
    random.shuffle(graphs)
    fold_size = len(graphs) // K
    folds = [graphs[i*fold_size:(i+1)*fold_size] for i in range(K)]
    # Handle remainder
    for i in range(len(graphs) - K*fold_size):
        folds[i].append(graphs[K*fold_size + i])

    in_dim = graphs[0].x.shape[1]
    fold_accs = []

    for fold_idx in range(K):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx+1}/{K}")
        print(f"{'='*50}")

        val_graphs = folds[fold_idx]
        train_graphs = [g for i in range(K) if i != fold_idx for g in folds[i]]
        print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE * 2, shuffle=False)

        # Class weights from training set
        all_labels = []
        for g in train_graphs:
            all_labels.extend(g.y.numpy().tolist())
        label_counts = collections.Counter(all_labels)
        total = sum(label_counts.values())
        class_weights = torch.zeros(NUM_CLASSES)
        for i in range(NUM_CLASSES):
            class_weights[i] = total / max(label_counts.get(i, 1), 1)
        class_weights = class_weights.to(DEVICE)

        def evaluate(loader):
            model.eval()
            correct, total_, loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(DEVICE)
                    out = model(batch)
                    loss = F.nll_loss(out, batch.y, weight=class_weights)
                    loss_sum += loss.item() * batch.num_graphs
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total_ += batch.y.size(0)
            return correct / total_, loss_sum / len(loader.dataset)

        model = FaceClassifier(in_dim, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, DROPOUT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_acc = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                out = model(batch)
                loss = F.nll_loss(out, batch.y, weight=class_weights)
                loss.backward()
                optimizer.step()
            scheduler.step()

            _, val_acc = evaluate(val_loader)[0], evaluate(val_loader)[0]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if epoch % 20 == 0:
                train_acc, _ = evaluate(train_loader)
                print(f"  Epoch {epoch:3d} | train={train_acc:.3f} val={val_acc:.3f}")

        fold_accs.append(best_val_acc)
        print(f"  Fold {fold_idx+1} best val acc: {best_val_acc:.4f}")

    print(f"\n{'='*50}")
    print(f"K-FOLD RESULTS (k={K})")
    print(f"{'='*50}")
    print(f"Fold accuracies: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean: {np.mean(fold_accs):.4f}")
    print(f"Std:  {np.std(fold_accs):.4f}")

    # Train final model on all data
    print(f"\nTraining final model on all {len(graphs)} graphs...")
    all_loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
    final_model = FaceClassifier(in_dim, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # Class weights for all data
    all_labels = []
    for g in graphs:
        all_labels.extend(g.y.numpy().tolist())
    lc = collections.Counter(all_labels)
    t = sum(lc.values())
    cw = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        cw[i] = t / max(lc.get(i, 1), 1)
    cw = cw.to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        final_model.train()
        for batch in all_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out = final_model(batch)
            loss = F.nll_loss(out, batch.y, weight=cw)
            loss.backward()
            opt.step()
        sched.step()

    torch.save(final_model.state_dict(), MODEL_DIR / "face_classifier.pt")
    model_size = os.path.getsize(MODEL_DIR / "face_classifier.pt")
    print(f"Final model saved: {model_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
