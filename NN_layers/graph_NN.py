import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import nn


class GCN(torch.nn.Module):
    def __init__(
        self, input_features, hidden_channels, output_classes, dropout, layers
    ):
        super().__init__()
        torch.manual_seed(12)
        self.first_conv = GCNConv(input_features, hidden_channels)
        self.conv = GCNConv(hidden_channels, hidden_channels)
        self.last_conv = GCNConv(hidden_channels, output_classes)
        self.dropout = dropout

        assert layers >= 2, "the minimum number of layers for the GCN is 2"
        self.layers = layers

    def forward(self, x, edge_index):

        x = self.first_conv(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.layers - 2):
            x = self.conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.last_conv(x, edge_index)
        return x


class GatedSwitchesGNN(torch.nn.Module):
    """Configurable GNN Layer

    TODO modify description
    Implements the Gated Graph ConvNet layer:
        h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
        sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
        e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
        where Aggr. is an aggregation function: sum/mean/max.

    References:
        - Joshi, C. K. (2019). Graph convolutional neural networks for the travelling salesman problem.
    """

    def __init__(
        self,
        hidden_dim,
        aggregation="sum",
        norm="batch",
        learn_norm=True,
        track_norm=False,
        gated=True,
    ):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
            gated: Whether to use edge gating (True/False)
        """
        super(GatedSwitchesGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        assert self.gated, "Use gating with GNN, pass the `--gated` flag"

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # TODO look at the norms, I don't know what it does
        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(
                hidden_dim, affine=learn_norm, track_running_stats=track_norm
            ),
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(
                hidden_dim, affine=learn_norm, track_running_stats=track_norm
            ),
        }.get(self.norm, None)

    def forward(self, h, e, A, S):
        """
        Args:
            h: Input node features (B x V x H)
            e: Input switch features (B x V x V x H)
            A: Graph adjacency matrices (B x V x V)
            S: Switch adjacency matrices (B x V x V)
        Returns:
            Updated node and edge features
        """
        batch_size, num_nodes, hidden_dim = h.shape
        h_in = h
        e_in = e

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H
        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H
        Bh = self.B(h)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        # Update node features
        h = Uh + self.aggregate(Vh, S, gates) + self.aggregate(Vh, A, gates=1)  # B x V x H

        # Normalize node features
        h = (
            self.norm_h(h.view(batch_size * num_nodes, hidden_dim)).view(
                batch_size, num_nodes, hidden_dim
            )
            if self.norm_h
            else h
        )

        # Normalize edge features
        e = (
            self.norm_e(e.view(batch_size * num_nodes * num_nodes, hidden_dim)).view(
                batch_size, num_nodes, num_nodes, hidden_dim
            )
            if self.norm_e
            else e
        )

        # Apply non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        h = h_in + h
        e = e_in + e

        return h, e

    def aggregate(self, Vh, graph, gates):
        """
        Args:
            Vh: Neighborhood features (B x V x V x H)
            graph: Graph adjacency or switch-adjacency matrices (B x V x V)
            gates: Edge gates (B x V x V x H)
        Returns:
            Aggregated neighborhood features (B x V x H)
        """
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H

        # Enforce graph structure through masking
        Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

        if self.aggregation == "mean":
            return torch.sum(Vh, dim=2) / torch.sum(1 - graph, dim=2).unsqueeze(
                -1
            ).type_as(Vh)

        elif self.aggregation == "max":
            return torch.max(Vh, dim=2)[0]

        else:
            return torch.sum(Vh, dim=2)
