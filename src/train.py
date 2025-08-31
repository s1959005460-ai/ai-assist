"""
train.py — single-run entrypoint used by ablation_runner.py

Example usage:
python src/train.py --config configs/experiment_template.yaml --output_dir experiments/results/run1
"""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import yaml

from src.differentiable_topo import BatchedPersistenceLandscape, AdaptiveTopoReg
from src.learner import GCN, create_optimizer, SimpleClient
from src.server import FederatedServer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # set seeds for reproducibility
    import random, numpy as np
    seed = cfg.get("seed", 1234)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # NOTE: This script uses toy placeholders for dataset/client generation.
    # User should replace dataset creation with real torch_geometric Data splits.

    # Create a simple model and clients (placeholder)
    input_dim = cfg.get("input_dim", 16)
    num_classes = cfg.get("num_classes", 3)
    global_model = GCN(input_dim, cfg.get("hidden", 16), num_classes)

    # Create N clients as copies (placeholder)
    clients = []
    for i in range(cfg.get("num_clients", 3)):
        client_model = GCN(input_dim, cfg.get("hidden", 16), num_classes)
        client_model.load_state_dict(global_model.state_dict())
        # data placeholder: user must replace with actual torch_geometric.data.Data
        data = None
        opt, pw = create_optimizer(client_model, lr=cfg.get("lr", 0.01), noise_multiplier=cfg.get("noise_multiplier", None))
        clients.append(SimpleClient(client_model, data, optimizer=opt, privacy_wrapper=pw))

    # Topo module
    base = BatchedPersistenceLandscape(n_layers=cfg.get("n_layers",3), resolution=cfg.get("resolution",80), method=cfg.get("method","triangle"))
    topo_module = AdaptiveTopoReg(base, privacy_wrapper=None)

    server = FederatedServer(global_model, clients, data=None, topo_module=topo_module, lambda_topo=cfg.get("lambda_topo", 0.1), noise_multiplier=cfg.get("noise_multiplier", None))
    start = time.time()
    history = server.train(num_epochs=cfg.get("epochs", 5))
    dur = time.time() - start

    # Save a simple results json (extend as needed)
    results = {"config": cfg, "history": history, "runtime": dur}
    json.dump(results, open(outdir / "results.json", "w"), indent=2)
    print("Saved results to", outdir / "results.json")

if __name__ == "__main__":
    main()
