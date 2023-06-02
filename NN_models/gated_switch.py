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
        x_input = data.x
        x, s = self.Encoder(data.x, data.A, data.S)  # B x N x F, B x N x N x F

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

        p_switch = self.switch_activation(SMLP_out[:, 0])

        if warm_start:
            topology = self.PhyR(p_switch.flatten(), n_switches)
        else:
            topology = p_switch.flatten()

        graph_topo = torch.ones((200, utils.M), device=self.device)
        graph_topo[:, -utils.numSwitches :] = topology.view((200, -1))

        # ps_flow = torch.zeros((x.shape[0], utils.M), device=self.device)
        ps_flow = SMLP_out[:, 1].view((x.shape[0], -1))
        vs_parent = torch.zeros((x.shape[0], utils.M), device=self.device)
        vs_parent[:, -utils.numSwitches :] = SMLP_out[:, 2].view((x.shape[0], -1))
        vs_child = torch.zeros((x.shape[0], utils.M), device=self.device)
        vs_child[:, -utils.numSwitches :] = SMLP_out[:, 3].view((x.shape[0], -1))
        qs_flow = SMLP_out[:, 4].view((x.shape[0], -1))

        nodes = torch.nonzero(data.A.triu())  # dim = (M-num_switch)*B x 3

        x_begin = x[nodes[:, 0], nodes[:, 1], :]
        x_end = x[nodes[:, 0], nodes[:, 2], :]
        x_g_extended = x_g[nodes[:, 0], :]

        CMLP_input = torch.cat((x_begin, x_end, x_g_extended), dim=1)
        CMLP_out = self.CMLP(
            CMLP_input
        )  # (M-num_switch)*B x 3, [Pflow, Vparent, Vchild]

        pc_flow = CMLP_out[:, 0].view((x.shape[0], -1))
        vc_parent = torch.zeros((x.shape[0], utils.M), device=self.device)
        vc_parent[:, : -utils.numSwitches] = CMLP_out[:, 1].view((x.shape[0], -1))
        vc_child = torch.zeros((x.shape[0], utils.M), device=self.device)
        vc_child[:, : -utils.numSwitches] = CMLP_out[:, 2].view((x.shape[0], -1))

        # # TODO check if correct
        # # if want to try max aggregation
        # v11 = torch.abs(utils.A) @ vc_parent.double().unsqueeze(2)
        # v12 = torch.abs(utils.A) @ vs_parent.double().unsqueeze(2)
        # v21 = torch.abs(utils.A) @ vc_child.double().unsqueeze(2)
        # v22 = torch.abs(utils.A) @ vs_child.double().unsqueeze(2)

        # tempv = torch.cat((v11, v12, v21, v22), dim=2)

        # v = torch.max(tempv, dim=2)[0].squeeze()

        v = (
            utils.D_inv
            @ utils.Incidence_parent
            @ (vc_parent + vs_parent).unsqueeze(2).double()
            + utils.D_inv
            @ utils.Incidence_child
            @ (vc_child + vs_child).unsqueeze(2).double()
        ).squeeze()
        v[:, 0] = 1  # V_PCC = 1

        pg, qg, q_flow_lines = utils.complete_JA(data.x, v, pc_flow, ps_flow, qs_flow)

        p_flow = torch.cat((pc_flow, ps_flow), dim=1)
        q_flow = torch.cat((q_flow_lines, qs_flow), dim=1)

        z = torch.cat((p_flow, v, graph_topo), dim=1)
        zc = torch.cat((q_flow, pg, qg), dim=1)

        # opt_z = data.y[:, : utils.zrdim]
        # opt_pij, opt_v, opt_topo = utils.decompose_vars_z_JA(opt_z)
        # opt_cpg, opt_cqg, _, opt_cqij = utils.complete_JA(
        #     data.x, opt_v, opt_pij, opt_topo
        # )

        # opt_zc = data.y[:, utils.zrdim :].double()
        # opt_qij, opt_pg, opt_qg = utils.decompose_vars_zc_JA(opt_zc)

        # opt_czc = torch.cat((opt_cqij, opt_cpg, opt_cqg), dim=1)

        # assert torch.allclose(opt_qij, opt_cqij, rtol=0, atol=1e-4), "qij not equal"
        # assert torch.allclose(opt_qg, opt_cqg, rtol=0, atol=1e-4), "qg not equal"
        # assert torch.allclose(opt_pg, opt_cpg, rtol=0, atol=1e-4), "pg not equal"

        return z, zc


class GatedSwitchGNN_globalMLP(nn.Module):
    def __init__(self, args, utils):
        super().__init__()
        output_dim = utils.M + utils.N + utils.numSwitches
        self.Encoder = GatedSwitchesEncoder(args)
        self.MLP = GlobalMLP_reduced_switch(
            args, utils.N + utils.numSwitches, output_dim
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

        # decode
        switches_nodes = torch.nonzero(data.S.triu())

        switches = s[
            switches_nodes[:, 0], switches_nodes[:, 1], switches_nodes[:, 2], :
        ]

        MLP_input = torch.cat((switches.view(200, -1), x.view(200, -1)), axis=1)
        MLP_out = self.MLP(MLP_input)  # [pij, v, p_switch]

        p_switch = self.switch_activation(MLP_out[:, -utils.numSwitches :])
        n_switch_per_batch = torch.full((200, 1), utils.numSwitches).squeeze()

        if warm_start:
            topology = self.PhyR(p_switch.flatten(), n_switch_per_batch)
        else:
            topology = p_switch.flatten()

        graph_topo = torch.ones((200, utils.M), device=self.device)
        graph_topo[:, -utils.numSwitches :] = topology.view((200, -1))

        v = MLP_out[:, utils.M : utils.M + utils.N]
        v[:, 0] = 1
        pg, qg, p_flow_corrected, q_flow_corrected = utils.complete_JA(
            data.x,
            v,
            MLP_out[:, 0 : utils.M],
            graph_topo,
            utils.A,
        )

        z = torch.cat((p_flow_corrected, v, graph_topo), dim=1)
        zc = torch.cat((q_flow_corrected, pg, qg), dim=1)
        return z, zc
