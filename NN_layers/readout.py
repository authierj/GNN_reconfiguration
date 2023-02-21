import torch
import torch.nn.functional as F
from torch.nn import Linear


class GlobalMLP(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_classes, layers, dropout):
        super().__init__()
        torch.manual_seed(12)
        self.lin1 = Linear(input_features, hidden_features)
        self.lin_middle = Linear(hidden_features, hidden_features)
        self.lin_last = Linear(hidden_features, output_classes)

        assert layers >= 1, "the minimum number of layers for an MLP is 1"
        self.layers = layers
        self.dropout = dropout

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        for k in range(self.layers-1):
            x = self.lin_middle(x)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_last(x)
        return x
