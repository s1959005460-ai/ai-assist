import json, glob
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob("experiment_results/*.json")
for f in files:
    data = json.load(open(f))
    if 'baseline_accs' not in data: continue
    runs = data['baseline_accs']
    mean, std = data['mean'], data['std']
    plt.bar(f.split('/')[-1], mean, yerr=std, capsize=5)
plt.ylabel("Test Accuracy (%)")
plt.title("Baseline mean±std over repeats")
plt.savefig("experiment_results/baseline_summary.png")
plt.show()
