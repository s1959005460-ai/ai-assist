# Experiments README

Folders:
- src/: core modules (learner, server, topo, train)
- tests/: unit tests (gradient numeric check)
- experiments/: ablation runner, visualization and results
- configs/: experiment template yaml

Quickstart:
1. Create venv
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows

2. Install requirements (adjust torch-geometric per your CUDA)
   pip install -r requirements.txt

3. Generate project files (already done by generate_all.py)

4. Run numeric gradient test
   pytest tests/test_gradient_correctness.py -q

5. Run visualization (optional)
   python experiments/viz_compare.py

6. Run a single training run (replace config with a real dataset config)
   python src/train.py --config configs/experiment_template.yaml --output_dir experiments/results/run1

7. Run ablation grid (calls train.py many times)
   python experiments/ablation_runner.py

Notes:
- Replace placeholder data logic in src/train.py with real torch_geometric Data and proper client split.
- Opacus attach in create_optimizer is a placeholder; adapt to the installed Opacus version by using DataLoader sample_rate or the proper attach API.
- Use experiment results in experiments/results/ for post-processing and plotting.
