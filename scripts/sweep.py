"""Hyperparameter sweep on fixed 2K samples. Runs overnight."""
import subprocess, json, time
from pathlib import Path
from datetime import datetime

ROOT = Path(r"D:\AiMeshGeoSegmenter")
PYTHON = r"C:\miniconda3\envs\occ\python.exe"
TRAIN = str(ROOT / "scripts" / "train.py")
RESULTS = ROOT / "models" / "sweep_results.json"

experiments = [
    # (name, feat, hidden, layers, heads, lr)
    ("v14_h192_l3",   "14", 192, 3, 4, 0.002),   # baseline
    ("v14_h256_l3",   "14", 256, 3, 4, 0.002),   # wider
    ("v14_h192_l4",   "14", 192, 4, 4, 0.002),   # deeper
    ("v14_h128_l3",   "14", 128, 3, 4, 0.002),   # compact
    ("v14_h128_l4",   "14", 128, 4, 4, 0.002),   # compact deeper
    ("v14_lr001",     "14", 192, 3, 4, 0.001),   # lower lr
    ("v14_lr003",     "14", 192, 3, 4, 0.003),   # higher lr
    ("v20_h192_l3",   "20", 192, 3, 4, 0.002),   # +fourier 6dim
    ("v20_h256_l3",   "20", 256, 3, 4, 0.002),
    ("v20_h192_l4",   "20", 192, 4, 4, 0.002),
    ("v26_h192_l3",   "26", 192, 3, 4, 0.002),   # +fourier 12dim
    ("v26_h256_l3",   "26", 256, 3, 4, 0.002),
    ("v26_h192_l4",   "26", 192, 4, 4, 0.002),
]

results = []
for i, (name, feat, hidden, layers, heads, lr) in enumerate(experiments):
    print(f"\n[{'='*50}]")
    print(f"[{i+1}/{len(experiments)}] {name}: feat={feat} h={hidden} L={layers} heads={heads} lr={lr}")
    t0 = time.time()

    cmd = [PYTHON, TRAIN, "--top2000", "--no_swa",
           "--feat", feat, "--hidden", str(hidden), "--layers", str(layers),
           "--heads", str(heads), "--lr", str(lr)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(ROOT))
        elapsed = time.time() - t0

        # Parse results from train_log.json
        log_json = ROOT / "models" / "train_log.json"
        if log_json.exists():
            d = json.loads(log_json.read_text())
            result = {
                "name": name, "feat": feat, "hidden": hidden, "layers": layers,
                "lr": lr, "best_val": d.get("best_val_acc"), "test_acc": d.get("test_acc"),
                "time_s": round(elapsed, 0), "timestamp": datetime.now().isoformat()
            }
        else:
            # Parse from train.log
            log_txt = ROOT / "models" / "train.log"
            text = log_txt.read_text(encoding="utf-8", errors="replace") if log_txt.exists() else ""
            bv = 0; ta = 0
            for line in text.split("\n"):
                if "Best Val Acc:" in line: bv = float(line.split(":")[-1].strip())
                if "Test Acc" in line: ta = float(line.split(":")[-1].strip().replace("(SWA)",""))
            result = {"name": name, "best_val": bv, "test_acc": ta, "time_s": round(elapsed, 0)}
    except subprocess.TimeoutExpired:
        result = {"name": name, "error": "timeout"}
    except Exception as e:
        result = {"name": name, "error": str(e)[:100]}

    results.append(result)
    print(f"  -> val={result.get('best_val','?')} test={result.get('test_acc','?')} {result.get('time_s','?')}s")
    with open(RESULTS, "w") as f:
        json.dump({"experiments": results, "updated": datetime.now().isoformat()}, f, indent=2)

print(f"\nDone. {len(results)} experiments.")
ranked = sorted([r for r in results if "best_val" in r], key=lambda r: r.get("best_val", 0), reverse=True)
print("\n=== RANKING ===")
for r in ranked:
    print(f"  {r['name']:16s} feat={r.get('feat','?'):4s} h={r.get('hidden','?')} L={r.get('layers','?')} val={r.get('best_val',0):.4f} test={r.get('test_acc',0):.4f} {r.get('time_s',0)}s")
