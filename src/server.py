import numpy as np
from src.learner import train_local, create_optimizer, SimpleClient

def fedavg(client_state_dicts, client_weights):
    new_state = {}
    total_weight = sum(client_weights)
    for k in client_state_dicts[0].keys():
        weighted_sum = sum(client_weights[i] * cs[k].float() for i, cs in enumerate(client_state_dicts))
        new_state[k] = weighted_sum / total_weight
    return new_state

class FederatedServer:
    """
    server.model: a model instance
    clients: list of SimpleClient instances
    data: server-side representative data (torch_geometric.data.Data)
    """
    def __init__(self, model, clients, data, topo_module=None, lambda_topo=0.1, noise_multiplier=None):
        self.model = model
        self.clients = clients
        self.data = data
        self.topo_module = topo_module
        self.lambda_topo = lambda_topo
        self.optimizer, self.privacy_wrapper = create_optimizer(self.model, noise_multiplier=noise_multiplier)
        self.global_topo_target = None

    def update_global_topo_target(self):
        if self.topo_module is None:
            self.global_topo_target = None
            return
        self.model.eval()
        with __import__("torch").no_grad():
            feats = self.model.get_hidden_features(self.data)
            node_feat = feats.unsqueeze(0)
            base = self.topo_module.base if hasattr(self.topo_module, 'base') else self.topo_module
            try:
                tgt = base(node_feat)
            except Exception:
                graph_feat = feats.mean(dim=0, keepdim=True)
                tgt = base(graph_feat)
            self.global_topo_target = tgt.detach().cpu()

    def recover_topo_consistency(self):
        # Placeholder: server-side mitigation strategy
        print("[Server] Recover topo consistency: lowering lambda or server-side fine-tune recommended.")

    def aggregate_state_dicts(self, client_state_dicts, client_weights):
        return fedavg(client_state_dicts, client_weights)

    def train(self, num_epochs=10):
        history = []
        if self.topo_module:
            self.update_global_topo_target()

        for epoch in range(num_epochs):
            client_state_dicts = []
            client_weights = []
            client_topos = []
            client_consistencies = []

            for client in self.clients:
                opt = client.optimizer if getattr(client, "optimizer", None) is not None else self.optimizer
                pw = client.privacy_wrapper if getattr(client, "privacy_wrapper", None) is not None else self.privacy_wrapper

                loss, topo_loss, consistency_loss = train_local(
                    client.model, client.data, opt,
                    privacy_wrapper=pw,
                    topo_module=self.topo_module,
                    topo_target=self.global_topo_target,
                    lambda_topo=self.lambda_topo,
                    lambda_consistency=0.5
                )

                client_state_dicts.append(client.model.state_dict())
                client_weights.append(1.0)
                client_topos.append(topo_loss)
                client_consistencies.append(consistency_loss)

            # Check consistency spread
            if len(client_consistencies) > 0 and self.topo_module is not None:
                std_consistency = float(np.std(client_consistencies))
                if std_consistency > 0.1:
                    print(f"[Server] Topo consistency std={std_consistency:.6f}, triggering recovery.")
                    self.recover_topo_consistency()

            # Aggregate
            agg = self.aggregate_state_dicts(client_state_dicts, client_weights)
            self.model.load_state_dict(agg)

            # Update global topo target after aggregation
            if self.topo_module:
                self.update_global_topo_target()

            history.append({
                "epoch": epoch,
                "mean_loss": np.mean([float(0 if c is None else 0) for c in client_state_dicts]), # placeholder
                "mean_topo": np.mean(client_topos) if client_topos else None,
                "mean_consistency": np.mean(client_consistencies) if client_consistencies else None
            })

        return history
