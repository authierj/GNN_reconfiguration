import torch
import torch.nn.functional as F
from torch.nn import Linear


class GlobalMLP(torch.nn.Module):
    def __init__(self, args, n_nodes, output_dim):
        super(GlobalMLP, self).__init__()
        # torch.manual_seed(12)
        input_features = args["outputFeatures"] * n_nodes
        hidden_features = input_features
        self.lin_input = Linear(input_features, hidden_features)
        self.lin_output = Linear(hidden_features, output_dim)
        self.dropout = args["dropout"]

    def forward(self, x):
        x = self.lin_input(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_output(x)
        return x


class GlobalMLP_reduced(torch.nn.Module):
    def __init__(self, args, n_nodes, output_dim):
        super(GlobalMLP_reduced, self).__init__()
        # torch.manual_seed(12)
        input_features = args["hiddenFeatures"] * n_nodes
        hidden_features = input_features
        self.lin_input = Linear(input_features, hidden_features)
        self.lin_output = Linear(hidden_features, output_dim)
        self.dropout = args["dropout"]

    def forward(self, x):
        x = self.lin_input(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_output(x)
        return x


class GlobalMLP_reduced_switch(torch.nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(GlobalMLP_reduced_switch, self).__init__()
        # torch.manual_seed(12)
        input_features = args["hiddenFeatures"] * input_dim
        hidden_features = input_features
        self.lin_input = Linear(input_features, hidden_features)
        self.lin_output = Linear(hidden_features, output_dim)
        self.dropout = args["dropout"]

    def forward(self, x):
        x = self.lin_input(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_output(x)
        return x


class SMLP(torch.nn.Module):
    def __init__(self, input_features, hidden_features, dropout):
        super(SMLP, self).__init__()
        # torch.manual_seed(12)
        self.lin_input = Linear(input_features, hidden_features)
        self.lin_output = Linear(hidden_features, 4)
        self.dropout = dropout

    def forward(self, x):
        x = self.lin_input(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_output(x)
        # x = x.sigmoid()
        return x


class CMLP(torch.nn.Module):
    def __init__(self, input_features, hidden_features, dropout):
        super(CMLP, self).__init__()
        # torch.manual_seed(12)
        self.lin_input = Linear(input_features, hidden_features)
        self.lin_output = Linear(hidden_features, 3)
        self.dropout = dropout

    def forward(self, x):
        x = self.lin_input(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_output(x)
        return x
