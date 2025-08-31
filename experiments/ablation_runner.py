"""
Ablation runner: runs train.py over a grid of hyperparameters and stores results.

Note: train.py in src/ must be adapted to real data; this runner calls it via subprocess.
"""
import subprocess, json, time, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "src" / "train.py"
RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

grid = {
    "lambda_topo": [0.0, 0.01, 0.1],
    "n_layers": [1,3],
    "method": ["triangle", "gaussian"],
    "noise_multiplier": [None, 0.5]
}

def run_one(cfg, run_id):
    outdir = RESULTS_DIR / f"exp_{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg_path = outdir / "config.json"
    json.dump(cfg, open(cfg_path, "w"))
    cmd = ["python", str(TRAIN_PY), "--config", str(cfg_path), "--output_dir", str(outdir)]
    print("Running:", " ".join(cmd))
    start = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.time() - start
    with open(outdir / "runner_stdout.txt", "w") as f:
        f.write(res.stdout + "\n\n" + res.stderr)
    print(f"Finished {run_id} in {dur:.1f}s; returncode {res.returncode}")

def main():
    all_cfgs = []
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    run_id = 0
    for vals in itertools.product(*values):
        cfg = dict(zip(keys, vals))
        # Map None to a serializable representation in config file
        cfg_serial = {k: (v if v is not None else None) for k,v in cfg.items()}
        # add other defaults
        cfg_serial.update({"epochs": 3, "num_clients": 3, "hidden": 16})
        run_one(cfg_serial, run_id)
        run_id += 1

if __name__ == "__main__":
    main()
