
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import torch
from torch_geometric.datasets import Planetoid
from server import FederatedServer
from learner import GCN
import asyncio
from typing import List

app = FastAPI()

# 加载数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# 初始化模型和服务器
model = GCN(dataset.num_features, 16, dataset.num_classes)
clients = [model] * 5  # 示例5个客户端
server = FederatedServer(model, clients, data)

class ConfigRequest(BaseModel):
    dataset: str
    model_type: str

# WebSocket 管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        for epoch in range(10):  # 示例循环
            # 每训练一轮，发送包含详细信息的JSON对象
            message = {
                "epoch": epoch,
                "loss": 0.5,  # 示例损失
                "accuracy": 0.85  # 示例准确率
            }
            await manager.send_message(json.dumps(message))
            await asyncio.sleep(1)  # 模拟训练过程
    except WebSocketDisconnect:
        manager.disconnect(websocket)
