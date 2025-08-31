"""
viz_compare.py
Compare approximate landscape produced by BatchedPersistenceLandscape with ripser/persim PD (if available).
Saves figures and numeric results to experiments/results/
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.differentiable_topo import BatchedPersistenceLandscape

try:
    from ripser import ripser
    from persim import PersLandscapeExact, bottleneck, wasserstein
    HAVE_RIPSER = True
except Exception:
    HAVE_RIPSER = False

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def synthetic_point_cloud(n=120, dim=2):
    rng = np.random.RandomState(0)
    return rng.normal(size=(n, dim))

def compute_true_pd(X):
    if not HAVE_RIPSER:
        return None
    return ripser(X, maxdim=1)["dgms"]

def diagram_to_landscape_np(diagrams, layer=3, res=100):
    if not HAVE_RIPSER:
        return None
    pl = PersLandscapeExact(dgms=diagrams, n_layers=layer, res=res)
    return pl.landscapes_

def approx_landscape(X_np, n_layers=3, res=100, method="triangle"):
    X = (X_np - X_np.min()) / (X_np.max() - X_np.min() + 1e-12)
    topo = BatchedPersistenceLandscape(n_layers=n_layers, resolution=res, method=method)
    with __import__("torch").no_grad():
        L = topo(__import__("torch").tensor(X, dtype=__import__("torch").float).unsqueeze(0))
    return L.squeeze(0).cpu().numpy()

def main():
    X = synthetic_point_cloud(150, 2)
    diagrams = compute_true_pd(X) if HAVE_RIPSER else None
    approx = approx_landscape(X, n_layers=3, res=100, method="triangle")

    results = {"have_ripser": HAVE_RIPSER}
    if diagrams is not None:
        true_land = diagram_to_landscape_np(diagrams, layer=3, res=100)
        # compute simple L2 between first layers
        l2 = float(((true_land[0] - approx[0])**2).sum()**0.5)
        results.update({"landscape_l2_first_layer": l2})
        # compute bottleneck/wasserstein on H1 if available
        try:
            bn = bottleneck(diagrams[1], diagrams[1])  # trivial: same diag (placeholder)
            ws = wasserstein(diagrams[1], diagrams[1])
            results.update({"bn_placeholder": float(bn), "ws_placeholder": float(ws)})
        except Exception:
            pass

        # plot
        plt.figure(figsize=(8,4))
        plt.plot(true_land[0], label="true_land[0]")
        plt.plot(approx[0], linestyle="--", label="approx[0]")
        plt.legend(); plt.title("Landscape layer 0: true vs approx")
        plt.savefig(RESULTS_DIR / "landscape_compare.png")
        plt.close()
    else:
        plt.figure(figsize=(6,3))
        plt.plot(approx[0], label="approx[0]")
        plt.legend()
        plt.savefig(RESULTS_DIR / "landscape_approx.png")
        plt.close()

    json.dump(results, open(RESULTS_DIR / "viz_results.json", "w"), indent=2)
    print("Saved viz results to", RESULTS_DIR)

if __name__ == "__main__":
    main()
