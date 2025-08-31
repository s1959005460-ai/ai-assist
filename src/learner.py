import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Optional Opacus import - user must adapt attach parameters per Opacus version
try:
    from opacus import PrivacyEngine
except Exception:
    PrivacyEngine = None

from torch_geometric.nn import GCNConv

# Import topo modules if present
try:
    from src.differentiable_topo import BatchedPersistenceLandscape, AdaptiveTopoReg
except Exception:
    # fallback import path if used as package
    try:
        from differentiable_topo import BatchedPersistenceLandscape, AdaptiveTopoReg
    except Exception:
        BatchedPersistenceLandscape = None
        AdaptiveTopoReg = None

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self._hidden = None

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        self._hidden = x
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_hidden_features(self, data):
        # Ensures hidden features are produced
        if self._hidden is None:
            _ = self.forward(data.x, data.edge_index)
        return self._hidden

class PrivacyWrapper:
    def __init__(self, privacy_engine=None, noise_multiplier=None, sample_rate=None):
        self.privacy_engine = privacy_engine
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate

    def step_post_backward(self):
        # Opacus-specific steps are usually handled on optimizer.step()
        return

def create_optimizer(model, lr=0.01, noise_multiplier=None, max_grad_norm=1.0, sample_rate=None):
    optimizer = Adam(model.parameters(), lr=lr)
    privacy_wrapper = None
    if noise_multiplier is not None and PrivacyEngine is not None:
        try:
            pe = PrivacyEngine()
            # NOTE: Opacus attach usage depends on version and requires a dataloader or sample_rate
            # User should adjust based on installed Opacus. This is a placeholder attach.
            # pe.attach(optimizer, sample_rate=sample_rate, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm)
            privacy_wrapper = PrivacyWrapper(privacy_engine=pe, noise_multiplier=noise_multiplier, sample_rate=sample_rate)
            privacy_wrapper.optimizer = optimizer
        except Exception:
            privacy_wrapper = PrivacyWrapper(privacy_engine=None, noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    else:
        privacy_wrapper = PrivacyWrapper(privacy_engine=None, noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return optimizer, privacy_wrapper

# Simple client wrapper
class SimpleClient:
    def __init__(self, model, data, optimizer=None, privacy_wrapper=None):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.privacy_wrapper = privacy_wrapper

def train_local(model, data, optimizer, privacy_wrapper=None,
                topo_module=None, topo_target=None,
                lambda_topo=0.1, lambda_consistency=0.5):
    """
    Train model on one local dataset (torch_geometric.data.Data)
    Returns: (total_loss, topo_self_loss, consistency_loss)
    """
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad()

    out = model(data.x.to(device), data.edge_index.to(device))
    ce_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    topo_self_loss = 0.0
    consistency_loss = 0.0

    if topo_module is not None:
        hidden = model.get_hidden_features(data).to(device)  # [N,H]
        node_feat = hidden.unsqueeze(0)  # [1,N,H]
        base = topo_module.base if hasattr(topo_module, 'base') else topo_module

        try:
            local_land = base(node_feat)
        except Exception:
            # fallback: pool to graph-level and compute
            graph_feat = hidden.mean(dim=0, keepdim=True)
            local_land = base(graph_feat)

        # self topo loss (norm vs zero)
        topo_self_loss = base.landscape_distance(local_land, torch.zeros_like(local_land)).mean()

        # consistency w.r.t topo_target (if provided)
        if topo_target is not None:
            tgt = topo_target.to(local_land.device)
            if tgt.dim() == 2:
                tgt = tgt.unsqueeze(0)
            consistency_loss = base.landscape_distance(local_land, tgt).mean()
        else:
            consistency_loss = torch.tensor(0.0, device=local_land.device)

        loss = ce_loss + lambda_topo * topo_self_loss + lambda_consistency * consistency_loss
    else:
        loss = ce_loss

    loss.backward()
    # DP hook - real Opacus step should be called via optimizer.step() or privacy_engine
    if privacy_wrapper is not None and hasattr(privacy_wrapper, 'privacy_engine'):
        # privacy_wrapper.privacy_engine.step() if that's the API - user must adapt
        pass
    elif privacy_wrapper is not None and hasattr(privacy_wrapper, 'step_post_backward'):
        privacy_wrapper.step_post_backward()

    optimizer.step()
    return float(loss.item()), float(topo_self_loss if not isinstance(topo_self_loss, torch.Tensor) else topo_self_loss.item()), float(consistency_loss if not isinstance(consistency_loss, torch.Tensor) else consistency_loss.item())
