import torch
import torch.multiprocessing as mp
from learner import Learner, GCN
from dataset import split_dataset

def client_process(client_id, model_class, data, device, topo_lambda, return_dict):
    learner = Learner(model_class(data.x.shape[1]), device=device)
    acc, t_loss = learner.train_local(data, epochs=1, lr=0.01, topo_lambda=topo_lambda)
    state_dict = {k: v.cpu() for k,v in learner.model.state_dict().items()}
    return_dict[client_id] = (state_dict, acc, t_loss)

class FederatedServer:
    def __init__(self, data, model_class=GCN, num_clients=3, devices=None, topo_lambda=0.0):
        self.num_clients = num_clients
        self.topo_lambda = topo_lambda
        self.client_data = split_dataset(data, num_clients=num_clients)
        if devices is None:
            devices = [f"cuda:{i}" for i in range(num_clients)]
        self.devices = devices
        self.model_class = model_class
        self.avg_acc_list, self.avg_topo_list = []

    def fedavg(self, client_states):
        """异步 FedAvg 聚合"""
        global_state = {}
        for name in client_states[0].keys():
            global_state[name] = sum(client_state[name] for client_state in client_states) / len(client_states)
        return global_state

    def train(self, rounds=5):
        manager = mp.Manager()
        for r in range(rounds):
            return_dict = manager.dict()
            processes = []
            for i, (data, device) in enumerate(zip(self.client_data, self.devices)):
                p = mp.Process(target=client_process, args=(i, self.model_class, data, device, self.topo_lambda, return_dict))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            client_states = []
            acc_list, topo_list = [], []
            for i in range(self.num_clients):
                state_dict, acc, t_loss = return_dict[i]
                client_states.append(state_dict)
                acc_list.append(acc)
                topo_list.append(t_loss)

            global_state = self.fedavg(client_states)

            # 更新客户端模型
            for i, device in enumerate(self.devices):
                learner = Learner(self.model_class(self.client_data[i].x.shape[1]), device=device)
                learner.model.load_state_dict(global_state)
                self.client_data[i].learner = learner

            avg_acc = sum(acc_list)/len(acc_list)
            avg_topo = sum(topo_list)/len(topo_list)
            self.avg_acc_list.append(avg_acc)
            self.avg_topo_list.append(avg_topo)
            print(f"[Round {r+1}/{rounds}] Avg Acc={avg_acc:.4f} | Avg Topo Loss={avg_topo:.4f}")

        print("[Server] Training Finished ✅")
