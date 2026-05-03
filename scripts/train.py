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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Train/val/test split
    random.seed(42)
    random.shuffle(graphs)
    n_train = int(len(graphs) * TRAIN_RATIO)
    n_val = int(len(graphs) * VAL_RATIO)
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train : n_train + n_val]
    test_graphs = graphs[n_train + n_val:]
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE * 2, shuffle=False)

    # Label stats
    all_labels = []
    for g in train_graphs:
        all_labels.extend(g.y.numpy().tolist())
    label_counts = collections.Counter(all_labels)
    total = sum(label_counts.values())
    print("Train label distribution:")
    for i, name in enumerate(LABEL_NAMES):
        c = label_counts.get(i, 0)
        print(f"  {name:12s}  {c:6d}  ({c/total*100:5.1f}%)")

    # Compute class weights for balanced loss
    class_weights = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        class_weights[i] = total / max(label_counts.get(i, 1), 1)
    class_weights = class_weights.to(DEVICE)

    # Model
    in_dim = graphs[0].x.shape[1]
    model = FaceClassifier(in_dim, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Device: {DEVICE}")
    print(f"Input dim: {in_dim}")

    def evaluate(loader):
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                loss = F.nll_loss(out, batch.y, weight=class_weights)
                loss_sum += loss.item() * batch.num_graphs
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        return correct / total, loss_sum / len(loader.dataset)

    # Training
    best_val_acc = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y, weight=class_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        scheduler.step()

        train_acc, _ = evaluate(train_loader)
        val_acc, val_loss = evaluate(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "face_classifier.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train acc={train_acc:.3f} | val acc={val_acc:.3f} | val loss={val_loss:.4f}")

    # Final test
    model.load_state_dict(torch.load(MODEL_DIR / "face_classifier.pt", map_location=DEVICE))
    test_acc, test_loss = evaluate(test_loader)
    print(f"\n=== Test Accuracy: {test_acc:.4f} ===")

    # Per-class test accuracy
    model.eval()
    class_correct = collections.Counter()
    class_total = collections.Counter()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            pred = out.argmax(dim=1)
            for i in range(batch.y.size(0)):
                gt = batch.y[i].item()
                class_total[gt] += 1
                if pred[i].item() == gt:
                    class_correct[gt] += 1

    print("\nPer-class accuracy:")
    for i, name in enumerate(LABEL_NAMES):
        c = class_correct.get(i, 0)
        t = class_total.get(i, 1)
        print(f"  {name:12s}  {c:5d}/{t:5d}  ({c/t*100:5.1f}%)")

    # Model size
    model_size = os.path.getsize(MODEL_DIR / "face_classifier.pt")
    print(f"\nModel saved: {model_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
