import torch
import torch.nn.functional as F
from torch.nn import Linear


class MLP(torch.nn.Module):
    def __init__(self, args, n_nodes, output_dim):
        super(MLP, self).__init__()
        # torch.manual_seed(12)
        input_features = 2*33
        hidden_features = 5
        self.lin_input = Linear(input_features, hidden_features)
        self.lin_hidden = Linear(hidden_features, hidden_features)
        self.lin_output = Linear(hidden_features, output_dim)
        self.dropout = args["dropout"]

    def forward(self, x):
        x = self.lin_input(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_output(x)
        return x