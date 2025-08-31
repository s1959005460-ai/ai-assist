import torch
from torch_geometric.datasets import Planetoid, KarateClub
from torch_geometric.data import Data

def load_dataset(name="KarateClub"):
    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=f"./data/{name}", name=name)
        return dataset
    elif name == "KarateClub":
        return [KarateClub()[0]]
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def split_dataset(data, num_clients=3):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    client_indices = [indices[i::num_clients] for i in range(num_clients)]
    client_data_list = []
    for idx in client_indices:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = True
        client_data = Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            y=data.y.clone(),
            train_mask=mask,
            val_mask=mask,
            test_mask=mask
        )
        client_data_list.append(client_data)
    return client_data_list
