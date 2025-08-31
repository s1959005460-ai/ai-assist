import torch
import torch.nn as nn

class BatchedPersistenceLandscape(nn.Module):
    """
    Vectorized, differentiable approximation of persistence landscapes.
    Input:
      X: Tensor [B, N, D] or [N, D]
    Output:
      landscapes: [B, n_layers, resolution]
    Methods: 'triangle' or 'gaussian'
    """
    def __init__(self, n_layers=3, resolution=100, eps=1e-8, method='triangle'):
        super().__init__()
        self.n_layers = n_layers
        self.resolution = resolution
        self.eps = eps
        self.method = method
        self.register_buffer('_x_grid', torch.linspace(0.0, 1.0, resolution))

    def forward(self, X):
        # normalize shape
        if X.dim() == 2:
            X = X.unsqueeze(0)
        B, N, D = X.shape
        device = X.device
        x = self._x_grid.to(device).view(1,1,-1)  # [1,1,R]

        # pairwise distances per graph
        dists = torch.cdist(X, X)  # [B,N,N]
        mean_dists = dists.mean(dim=2)  # [B,N]
        std_dists = dists.std(dim=2)    # [B,N]

        birth = mean_dists - std_dists
        death = mean_dists + std_dists
        lifetime = (death - birth).clamp_min(self.eps)

        k = min(self.n_layers, N)
        if k == 0:
            return torch.zeros(B, self.n_layers, self.resolution, device=device)

        _, idx = torch.topk(lifetime, k=k, dim=1)  # [B,k]
        selected_birth = torch.gather(birth, 1, idx)  # [B,k]
        selected_death = torch.gather(death, 1, idx)  # [B,k]

        if self.method == 'triangle':
            lands = self._triangle(x, selected_birth, selected_death)
        else:
            lands = self._gaussian(x, selected_birth, selected_death)

        if k < self.n_layers:
            pad = torch.zeros(B, self.n_layers - k, self.resolution, device=device)
            lands = torch.cat([lands, pad], dim=1)
        return lands  # [B, n_layers, R]

    def _triangle(self, x, birth, death):
        mid = (birth + death) / 2.0
        b = birth.unsqueeze(-1); m = mid.unsqueeze(-1); d = death.unsqueeze(-1)
        denom_left = (m - b).clamp_min(self.eps)
        left = ((x - b) / denom_left).clamp(0.0, 1.0)
        denom_right = (d - m).clamp_min(self.eps)
        right = ((d - x) / denom_right).clamp(0.0, 1.0)
        tri = torch.where(x <= m, left, right)  # [B,k,R]
        return tri

    def _gaussian(self, x, birth, death):
        mid = (birth + death) / 2.0
        sigma = ((death - birth).clamp_min(self.eps) / 4.0).unsqueeze(-1)
        m = mid.unsqueeze(-1)
        gauss = torch.exp(-0.5 * ((x - m) / sigma).pow(2))
        return gauss

    def to_vector(self, landscapes):
        # [B, L, R] -> [B, L*R]
        return landscapes.flatten(start_dim=1)

    def landscape_distance(self, a, b, metric='l2'):
        va = self.to_vector(a); vb = self.to_vector(b)
        if metric == 'l2':
            return ((va - vb).pow(2).sum(dim=1)).sqrt()
        elif metric == 'cosine':
            return 1 - torch.nn.functional.cosine_similarity(va, vb, dim=1)
        else:
            raise ValueError("Unknown metric")

class AdaptiveTopoReg(nn.Module):
    """
    Wraps a base BatchedPersistenceLandscape and scales loss based on privacy_wrapper.noise_multiplier if provided.
    """
    def __init__(self, base_module, privacy_wrapper=None):
        super().__init__()
        self.base = base_module
        self.privacy_wrapper = privacy_wrapper

    def forward(self, X):
        # Accept node-level [N,H], graph-level [1,H] or batched [B,N,H]
        if X.dim() == 2:
            # treat as graph-level feature attempt
            try:
                landscapes = self.base(X.unsqueeze(0))
            except Exception:
                landscapes = self.base(X.unsqueeze(0))
        else:
            landscapes = self.base(X)
        topo_loss = self.base.landscape_distance(landscapes, torch.zeros_like(landscapes)).mean()
        if self.privacy_wrapper is not None:
            nm = getattr(self.privacy_wrapper, "noise_multiplier", None)
            if nm is not None:
                scale = 1.0 / (1.0 + float(nm))
                return topo_loss * scale
        return topo_loss
