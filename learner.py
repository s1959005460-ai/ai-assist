import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from ripser import ripser

class GCN(nn.Module):
    def __init__(self, in_feats, hidden=16, out_feats=2):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, out_feats)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def topo_loss(x, device="cuda"):
    """GPU 全流程 topo_loss"""
    x_np = x.detach().cpu().numpy()
    diagrams = ripser(x_np, maxdim=1)['dgms']
    loss = torch.tensor(0.0, device=device)
    for dgm in diagrams:
        if len(dgm) > 0:
            pts = torch.tensor(dgm, device=device, dtype=torch.float32)
            loss += torch.mean((pts[:,1] - pts[:,0])**2)
    return loss

class Learner:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device

    def train_local(self, data, epochs=1, lr=0.01, topo_lambda=0.0):
        data = data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            if topo_lambda > 0:
                loss += topo_lambda * topo_loss(out, device=self.device)
            loss.backward()
            optimizer.step()
        acc = self.evaluate(data)
        t_loss = topo_loss(self.model(data.x, data.edge_index), device=self.device)
        return acc, t_loss.item()

    def evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[data.train_mask] == data.y[data.train_mask]).sum().item()
            acc = correct / int(data.train_mask.sum().item())
        self.model.train()
        return acc
