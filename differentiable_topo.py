# differentiable_topo.py
import torch

def compute_landscape_approx(model, data):
    """
    占位符：近似 persistence landscape 计算。
    当前返回 0.0，不影响 baseline 训练。
    未来可替换为实际的 differentiable topology 正则项。
    """
    return torch.tensor(0.0, device=next(model.parameters()).device)
