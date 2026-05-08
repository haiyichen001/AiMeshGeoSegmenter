"""Hyperparameter sweep on fixed 2K samples. Runs overnight."""
import sys, subprocess, json, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
PYTHON = sys.executable
TRAIN = str(ROOT / "scripts" / "train.py")
RESULTS = ROOT / "models" / "sweep_results.json"

experiments = [
    # (name, feat, hidden, layers, heads, lr)
    ("baseline",      "14", 192, 3, 4, 0.002),   # baseline for comparison
    ("v26_base",      "26", 192, 3, 4, 0.002),   # best from v1 sweep
    ("v26_jk",        "26", 192, 3, 4, 0.002),   # JK (already on by default now)
    ("v26_deep",      "26", 192, 4, 4, 0.002),   # deeper + JK
    ("v26_deep256",   "26", 256, 4, 4, 0.002),   # wider + deeper + JK
    ("v26_amp_wide",  "26", 320, 3, 4, 0.002),   # AMP enables bigger hidden
    ("v26_amp_deep",  "26", 256, 4, 4, 0.002),   # deeper with AMP
    ("v14_jk",        "14", 192, 3, 4, 0.002),   # JK on baseline
    ("v14_jk_deep",   "14", 192, 4, 4, 0.002),   # JK + deeper
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
