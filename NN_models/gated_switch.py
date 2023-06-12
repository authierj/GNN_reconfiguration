import torch
import torch.nn as nn
from NN_layers.graph_NN import GatedSwitchesLayer, FirstGatedSwitchesLayer
from NN_layers.readout import *
from utils_JA import xgraph_xflatten, Modified_Sigmoid


class GatedSwitchesEncoder(nn.Module):
    """Configurable GNN Encoder"""

    def __init__(self, args, learn_norm=False, track_norm=False, **kwargs):
        super(GatedSwitchesEncoder, self).__init__()

        self.init_embed_edges = nn.Embedding(2, args["inputFeatures"])

        layers = [
            GatedSwitchesLayer(
                args["hiddenFeatures"],
                args["aggregation"],
                args["norm"],
                learn_norm,
                track_norm,
            )
            for _ in range(args["numLayers"] - 1)
        ]
        layers.insert(
            0,
            FirstGatedSwitchesLayer(
                args["inputFeatures"],
                args["hiddenFeatures"],
                args["aggregation"],
                args["norm"],
                learn_norm,
                track_norm,
            ),
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x, A, S):
        """
        Args:
            x: Input node features (B x V x H)
            ei: Graph adjacency matrices (B x 2 x M-numSwitches)
            si: Switch adjacency matrices (B x 2 x numSwitches)
        Returns:
            Updated node features (B x V x H)
            Updated switch features (B x V x V x H)
        """
        # Embed switch features
        s = self.init_embed_edges(S.to_dense().type(torch.long))

        for layer in self.layers:
            x, s = layer(x, s, A, S)

        return x, s


class GatedSwitchGNN(nn.Module):
    def __init__(self, args, utils):
        super().__init__()
        self.Encoder = GatedSwitchesEncoder(args)
        self.SMLP = SMLP(
            4 * args["hiddenFeatures"], 4 * args["hiddenFeatures"], args["dropout"]
        )
        self.CMLP = CMLP(
            3 * args["hiddenFeatures"], 3 * args["hiddenFeatures"], args["dropout"]
        )
        self.device = args["device"]
        self.PhyR = getattr(utils, args["PhyR"])
        if args["switchActivation"] == "sig":
            self.switch_activation = nn.Sigmoid()
        elif args["switchActivation"] == "mod_sig":
            self.switch_activation = Modified_Sigmoid()
        else:
            self.switch_activation = nn.Identity()

    def forward(self, data, utils, warm_start=False):
        # encode
        x, s = self.Encoder(data.x, data.A, data.S)  # B x N x F, B x N x N x F
        mask = (
            torch.arange(utils.M, device=utils.device).unsqueeze(0)
            >= utils.M - data.numSwitches
        )

        # decode
        switches_nodes = torch.nonzero(data.S.triu())

        switches = s[
            switches_nodes[:, 0], switches_nodes[:, 1], switches_nodes[:, 2], :
        ]
        x_1 = x[switches_nodes[:, 0], switches_nodes[:, 1], :]
        x_2 = x[switches_nodes[:, 0], switches_nodes[:, 2], :]

        x_g = torch.sum(x, dim=1)  # B x F
        x_g_extended = x_g[switches_nodes[:, 0], :]  # dim = num_switches*B x F

        SMLP_input = torch.cat(
            (switches, x_1, x_2, x_g_extended), dim=1
        )  # num_switches*B x 4F
        SMLP_out = self.SMLP(
            SMLP_input
        )  # num_switches*B x 4, [switch_prob, P_flow, V_parent, V_child]

        p_switch = SMLP_out[:, 0]
        graph_topo = torch.ones((x.shape[0], utils.M), device=self.device)
        graph_topo[mask] = p_switch

        if warm_start:
            graph_topo = self.PhyR(graph_topo)

        ps_flow = torch.zeros((x.shape[0], utils.M), device=self.device)
        ps_flow[mask] = SMLP_out[:, 1] - 0.5
        vs_parent = torch.zeros((x.shape[0], utils.M), device=self.device)
        vs_parent[mask] = utils.vLow * (1 - SMLP_out[:, 2]) + utils.vUpp * (
            SMLP_out[:, 2]
        )
        vs_child = torch.zeros((x.shape[0], utils.M), device=self.device)
        vs_child[mask] = utils.vLow * (1 - SMLP_out[:, 3]) + utils.vUpp * (SMLP_out[:, 3])

        nodes = torch.nonzero(data.A.triu())  # dim = (M-num_switch)*B x 3

        x_begin = x[nodes[:, 0], nodes[:, 1], :]
        x_end = x[nodes[:, 0], nodes[:, 2], :]
        x_g_extended = x_g[nodes[:, 0], :]

        CMLP_input = torch.cat((x_begin, x_end, x_g_extended), dim=1)
        CMLP_out = self.CMLP(
            CMLP_input
        )  # (M-num_switch)*B x 3, [Pflow, Vparent, Vchild]

        pc_flow = torch.zeros((x.shape[0], utils.M), device=self.device)
        pc_flow[~mask] = CMLP_out[:, 0] - 0.5
        vc_parent = torch.zeros((x.shape[0], utils.M), device=self.device)
        vc_parent[~mask] = utils.vLow * (1 - CMLP_out[:, 1]) + utils.vUpp * (CMLP_out[:, 1])
        vc_child = torch.zeros((x.shape[0], utils.M), device=self.device)
        vc_child[~mask] = utils.vLow * (1 - CMLP_out[:, 2]) + utils.vUpp * (CMLP_out[:, 2])

        p_flow = ps_flow + pc_flow
        # first_mul = data.D_inv @ data.Incidence_parent
        # second_mul = first_mul @ Vc_parent.unsqueeze(2).double()
        B, N = data.inv_degree.shape
        D_inv = torch.zeros((B, N, N), device=utils.device)
        D_inv[
            :,
            torch.arange(N, device=utils.device),
            torch.arange(N, device=utils.device),
        ] = data.inv_degree

        v = (
            D_inv
            @ data.inc_parents.float()
            @ (vc_parent + vs_parent).unsqueeze(2).float()
            + D_inv
            @ data.inc_childs.float()
            @ (vc_child + vs_child).unsqueeze(2).float()
        ).squeeze()
        v[:, 0] = 1  # V_PCC = 1

        pg, qg, p_flow_corrected, q_flow_corrected = utils.complete_JA(
            data.x_data, v, p_flow, graph_topo, data.incidence, data.Rall, data.Xall
        )

        z = torch.cat((p_flow_corrected, v, graph_topo), dim=1)
        zc = torch.cat((q_flow_corrected, pg, qg), dim=1)

        return z, zc


class GatedSwitchGNN_globalMLP(nn.Module):
    def __init__(self, args, utils):
        super().__init__()
        output_dim = utils.M + utils.N + utils.numSwitches
        self.Encoder = GatedSwitchesEncoder(args)
        self.MLP = GlobalMLP_reduced_switch(args, utils.N, output_dim)
        self.device = args["device"]
        self.PhyR = getattr(utils, args["PhyR"])
        if args["switchActivation"] == "sig":
            self.switch_activation = nn.Sigmoid()
        elif args["switchActivation"] == "mod_sig":
            self.switch_activation = Modified_Sigmoid()
        else:
            self.switch_activation = nn.Identity()

    def forward(self, data, utils, warm_start=False):
        # encode
        x, s = self.Encoder(data.x, data.A, data.S)  # B x N x F, B x N x N x F
        mask = (
            torch.arange(utils.M, device=utils.device).unsqueeze(0)
            >= utils.M - data.numSwitches
        )

        # decode
        switches_nodes = torch.nonzero(data.S.triu())

        switches = s[
            switches_nodes[:, 0], switches_nodes[:, 1], switches_nodes[:, 2], :
        ]

        SMLP_input = torch.cat((switches.view(200, -1), x.view(200, -1)), axis=1)
        SMLP_out = self.MLP(SMLP_input)  # [pij, v, p_switch]

        SMLP_input = torch.cat((switches.view(200, -1), x.view(200, -1)), axis=1)
        SMLP_out = self.MLP(SMLP_input)  # [pij, v, p_switch]

        p_switch = self.switch_activation(SMLP_out[:, -utils.numSwitches : :])
        graph_topo = torch.ones((x.shape[0], utils.M), device=self.device)
        graph_topo[mask] = p_switch

        if warm_start:
            graph_topo = self.PhyR(graph_topo, data.numSwitches)

        v = SMLP_out[:, utils.M : utils.M + utils.N]
        v[:, 0] = 1
        pg, qg, p_flow_corrected, q_flow_corrected = utils.complete_JA(
            data.x_mod,
            v,
            SMLP_out[:, 0 : utils.M],
            graph_topo,
            utils.A,
        )

        z = torch.cat((p_flow_corrected, v, graph_topo), dim=1)
        zc = torch.cat((q_flow_corrected, pg, qg), dim=1)
        return z, zc
