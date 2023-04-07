import torch
import torch.nn as nn
from NN_layers.graph_NN import GatedSwitchesLayer, FirstGatedSwitchesLayer
from NN_layers.readout import *


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
                args["gated"],
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
                args["gated"],
            ),
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x, A, S):
        """
        Args:
            x: Input node features (B x V x H)
            A: Graph adjacency matrices (B x V x V)
            S: Switch adjacency matrices (B x V x V)
        Returns:
            Updated node features (B x V x H)
            Updated switch features (B x V x V x H)
        """
        # Embed switch features
        s = self.init_embed_edges(S.type(torch.long))

        # print(self.layers)

        for layer in self.layers:
            x, s = layer(x, s, A, S)

        return x, s


class GatedSwitchGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.Encoder = GatedSwitchesEncoder(args)
        self.SMLP = SMLP(
        4 * args["hiddenFeatures"], 4 * args["hiddenFeatures"], args["dropout"]
        )
        self.CMLP = CMLP(
            3 * args["hiddenFeatures"], 3 * args["hiddenFeatures"], args["dropout"]
        )
        self.completion_step = args["useCompl"]

    def forward(self, data, utils):

        # encode
        x, s = self.Encoder(data.x_mod, data.A, data.S)  # B x N x F, B x N x N x F
        
        test_switch = s[0,:,:,0] #problem here!
        # decode
        switches_nodes = torch.nonzero(data.S.triu())
        n_switches = torch.sum(torch.sum(data.S, dim=1), dim=1) // 2

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
        topology = utils.physic_informed_rounding(
            SMLP_out[:, 0], n_switches
        )  # num_switches*B
        graph_topo = torch.ones((x.shape[0], utils.M)).bool()
        graph_topo[data.switch_mask] = topology

        ps_flow = torch.zeros((x.shape[0], utils.M))
        ps_flow[data.switch_mask] = SMLP_out[:, 1]
        vs_parent = torch.zeros((x.shape[0], utils.M))
        vs_parent[data.switch_mask] = SMLP_out[:, 2]
        vs_child = torch.zeros((x.shape[0], utils.M))
        vs_child[data.switch_mask] = SMLP_out[:, 3]

        nodes = torch.nonzero(data.A.triu())  # dim = (M-num_switch)*B x 3

        x_begin = x[nodes[:, 0], nodes[:, 1], :]
        x_end = x[nodes[:, 0], nodes[:, 2], :]
        x_g_extended = x_g[nodes[:, 0], :]

        CMLP_input = torch.cat((x_begin, x_end, x_g_extended), dim=1)
        CMLP_out = self.CMLP(
            CMLP_input
        )  # (M-num_switch)*B x 3, [Pflow, Vparent, Vchild]

        pc_flow = torch.zeros((x.shape[0], utils.M))
        pc_flow[~data.switch_mask] = CMLP_out[:, 0]
        vc_parent = torch.zeros((x.shape[0], utils.M))
        vc_parent[~data.switch_mask] = CMLP_out[:, 1]
        vc_child = torch.zeros((x.shape[0], utils.M))
        vc_child[~data.switch_mask] = CMLP_out[:, 2]

        p_flow = ps_flow + pc_flow
        # first_mul = data.D_inv @ data.Incidence_parent
        # second_mul = first_mul @ Vc_parent.unsqueeze(2).double()
        v = (
            data.D_inv @ data.Incidence_parent.float() @ (vc_parent + vs_parent).unsqueeze(2).float()
            + data.D_inv @ data.Incidence_child.float() @ (vc_child + vs_child).unsqueeze(2).float()
        ).squeeze()
        v[:, 0] = 1  # V_PCC = 1

        pg, qg, p_flow_corrected, q_flow_corrected = utils.complete_JA(
            data.x_mod, v, p_flow, graph_topo, data.Incidence
        )

        z = torch.cat((p_flow_corrected, v, graph_topo), dim=1)
        zc = torch.cat((q_flow_corrected, pg, qg), dim=1)

        # p_test = torch.sum(data.x_mod[:,:,0], dim=1) - torch.sum(pg, dim=1)
        # q_test = torch.sum(data.x_mod[:,:,1], dim=1) - torch.sum(qg, dim=1)
        return z, zc


class GatedSwitchGNN_globalMLP(nn.Module):
    def __init__(self, args, N, output_dim):
        super().__init__()
        self.Encoder = GatedSwitchesEncoder(args)
        self.MLP = GlobalMLP_reduced(args, N, output_dim)
        self.completion_step = args["useCompl"]

    def forward(self, data, utils):

        # encode
        x, s = self.Encoder(data.x_mod, data.A, data.S)  # B x N x F, B x N x N x F
        
        # decode
        switches_nodes = torch.nonzero(data.S.triu())
        n_switches = torch.sum(torch.sum(data.S, dim=1), dim=1) // 2

        switches = s[
            switches_nodes[:, 0], switches_nodes[:, 1], switches_nodes[:, 2], :
        ]
        
        SMLP_input = torch.cat((switches.view(200,-1), x.view(200, -1)), axis=1)
        SMLP_out = self.MLP(SMLP_input) #[pij, v, p_switch]
        
        p_switch = SMLP_out[:, -utils.numSwitches : :]
        n_switch_per_batch = torch.full((200, 1), utils.numSwitches).squeeze()

        topology = utils.physic_informed_rounding(
            p_switch.flatten(), n_switch_per_batch
        )
        graph_topo = torch.ones((200, utils.M)).bool()
        graph_topo[:, -utils.numSwitches : :] = topology.view((200, -1))

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