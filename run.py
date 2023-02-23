# Extern
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from torch_geometric.loader import DataLoader


# Local
from NN_layers.graph_NN import GCN
from utils_JA import Utils
from utils_JA import xgraph_xflatten
from NN_layers.readout import GlobalMLP
from datasets.graphdataset import GraphDataSet
import default_args


class NeuralNet(nn.Module):
    def __init__(self, GNN, readout):
        super().__init__()
        self.GNN = GNN
        self.readout = readout

    def forward(self, x, graph, training):
        x = self.GNN(x, graph)
        if training:
            x_nn = xgraph_xflatten(x, 200, first_node=True)
        else:
            x_nn = xgraph_xflatten(x, 400, first_node=True)
        x = self.readout(x_nn)

        return x


filepath = 'datasets/node4/processed/node4_dataset'
batch_size = 200
try:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print('Network file does not exist')

args = default_args.method_default_args()
utils = Utils(data)

graph_dataset = GraphDataSet(root='datasets/node4')
train_graphs = graph_dataset[0:3200]
valid_graphs = graph_dataset[3200:3600]
test_graphs = graph_dataset[3600:4000]

train_loader = DataLoader(train_graphs, batch_size=200, shuffle=True)
valid_loader = DataLoader(
    valid_graphs, batch_size=len(valid_graphs), shuffle=True)
test_loader = DataLoader(
    test_graphs, batch_size=len(test_graphs), shuffle=True)

hidden_features_GNN = 12
output_features_GNN = 12
GNN = GCN(input_features=2, hidden_channels=hidden_features_GNN,
          output_classes=output_features_GNN, layers=4)

hidden_features_MLP = 12
readout = GlobalMLP(input_features=output_features_GNN*data.M,
                    hidden_channels=hidden_features_MLP*data.M,
                    output_classes=data.zdim, layers=2, dropout=0.5)

model = NeuralNet(GNN, readout)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01, weight_decay=5e-4)
cost_fnc = utils.obj_fnc

for epoch in range(1, 100):
    loss = train(model, optimizer, cost_fnc, train_loader, args, utils)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    eval_loss = test_or_validate(
        model, cost_fnc, valid_loader, args, utils)
    print(f'Eval_Loss: {eval_loss:.4f}')