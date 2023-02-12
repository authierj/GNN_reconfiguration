import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, output_classes, layers):
        super().__init__()
        torch.manual_seed(12)
        self.first_conv = GCNConv(input_features, hidden_channels)
        self.conv = GCNConv(hidden_channels, hidden_channels)
        self.last_conv = GCNConv(hidden_channels, output_classes)

        assert layers >= 2, "the minimum number of layers for the GCN is 2"
        self.layers = layers

    def forward(self, x, edge_index):

        x = self.first_conv(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        for i in range(self.layers-2):
            x = self.conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.last_conv(x, edge_index)
        return x
