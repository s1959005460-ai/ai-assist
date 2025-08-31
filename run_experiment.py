import torch
import matplotlib.pyplot as plt
from dataset import load_dataset
from server import FederatedServer

def main():
    dataset_name = "Cora"
    topo_lambda = 0.01
    repeats = 1
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]

    dataset = load_dataset(dataset_name)
    data = dataset[0] if not isinstance(dataset, list) else dataset[0]

    for r in range(repeats):
        print(f"==== Run {r+1}/{repeats} on {dataset_name} ====")
        server = FederatedServer(data, num_clients=len(devices), devices=devices, topo_lambda=topo_lambda)
        server.train(rounds=5)

        plt.figure(figsize=(6,4))
        plt.plot(server.avg_acc_list, label="Avg Accuracy")
        plt.plot(server.avg_topo_list, label="Avg Topo Loss")
        plt.xlabel("Round")
        plt.ylabel("Value")
        plt.title(f"{dataset_name} Async Multi-GPU FedAvg + TopoReg")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
