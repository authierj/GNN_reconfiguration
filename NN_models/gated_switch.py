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

        # p_switch = self.switch_activation(SMLP_out[:, 0])
        p_switch = SMLP_out[:, 0]

        if warm_start:
            topology = self.PhyR(p_switch.flatten(), n_switches)
        else:
            topology = p_switch.flatten()

        graph_topo = torch.ones((200, utils.M), device=self.device)
        graph_topo[:, -utils.numSwitches :] = topology.view((200, -1))

        ps_flow = torch.zeros((x.shape[0], utils.M), device=self.device)
        ps_flow[:, -utils.numSwitches :] = SMLP_out[:, 1].view((x.shape[0], -1)) - 0.5
        vs_parent = torch.zeros((x.shape[0], utils.M), device=self.device)
        vs_parent[:, -utils.numSwitches :] = utils.vLow * (1 - SMLP_out[:, 2].view((x.shape[0], -1))) + utils.vUpp * SMLP_out[:, 2].view((x.shape[0], -1))
        vs_child = torch.zeros((x.shape[0], utils.M), device=self.device)
        vs_child[:, -utils.numSwitches :] = utils.vLow * (1 - SMLP_out[:, 3].view((x.shape[0], -1))) + utils.vUpp * SMLP_out[:, 3].view((x.shape[0], -1))

        nodes = torch.nonzero(data.A.triu())  # dim = (M-num_switch)*B x 3

        x_begin = x[nodes[:, 0], nodes[:, 1], :]
        x_end = x[nodes[:, 0], nodes[:, 2], :]
        x_g_extended = x_g[nodes[:, 0], :]

        CMLP_input = torch.cat((x_begin, x_end, x_g_extended), dim=1)
        CMLP_out = self.CMLP(
            CMLP_input
        )  # (M-num_switch)*B x 3, [Pflow, Vparent, Vchild]

        pc_flow = torch.zeros((x.shape[0], utils.M), device=self.device)
        pc_flow[:, : -utils.numSwitches] = CMLP_out[:, 0].view((x.shape[0], -1)) - 0.5
        vc_parent = torch.zeros((x.shape[0], utils.M), device=self.device)
        vc_parent[:, : -utils.numSwitches] = utils.vLow * (1 - CMLP_out[:, 1].view((x.shape[0], -1))) + utils.vUpp * CMLP_out[:, 1].view((x.shape[0], -1))
        vc_child = torch.zeros((x.shape[0], utils.M), device=self.device)
        vc_child[:, : -utils.numSwitches] = utils.vLow * (1 - CMLP_out[:, 2].view((x.shape[0], -1))) + utils.vUpp * CMLP_out[:, 2].view((x.shape[0], -1))

        p_flow = ps_flow + pc_flow

        # TODO check if correct
        # if want to try max aggregation

        # assert torch.allclose(utils.Incidence_parent - utils.Incidence_child, utils.A)

        # v11 = utils.Incidence_parent @ vc_parent.double().unsqueeze(2)
        # v12 = utils.Incidence_parent @ vs_parent.double().unsqueeze(2)
        # v21 = utils.Incidence_child @ vc_child.double().unsqueeze(2)
        # v22 = utils.Incidence_child @ vs_child.double().unsqueeze(2)

        # v11_0 = utils.Incidence_parent @ vc_parent[0, :].double()
        # v12_0 = utils.Incidence_parent @ vs_parent[0, :].double()
        # v21_0 = utils.Incidence_parent @ vc_child[0, :].double()
        # v22_0 = utils.Incidence_parent @ vs_child[0, :].double()

        # tempv = torch.cat((v21, v22), dim=2)
        # tempv0 = torch.cat(
        #     (
        #         v11_0.unsqueeze(1),
        #         v12_0.unsqueeze(1),
        #         v21_0.unsqueeze(1),
        #         v22_0.unsqueeze(1),
        #     ),
        #     dim=1,
        # )

        # v = torch.max(tempv, dim=2)[0].squeeze()
        # v0 = torch.max(tempv0, dim=1)[0].squeeze()

        v = (
            utils.D_inv
            @ utils.Incidence_parent
            @ (vc_parent + vs_parent).unsqueeze(2).double()
            + utils.D_inv
            @ utils.Incidence_child
            @ (vc_child + vs_child).unsqueeze(2).double()
        ).squeeze()
        v[:, 0] = 1  # V_PCC = 1

        pg, qg, p_flow_corrected, q_flow_corrected = utils.complete_JA(
            data.x_data, v, p_flow, graph_topo
        )

        z = torch.cat((p_flow_corrected, v, graph_topo), dim=1)
        zc = torch.cat((q_flow_corrected, pg, qg), dim=1)

        opt_z = data.y[:, : utils.zrdim]
        opt_pij, opt_v, opt_topo = utils.decompose_vars_z_JA(opt_z)
        opt_cpg, opt_cqg, _, opt_cqij = utils.complete_JA(
            data.x_data, opt_v, opt_pij, opt_topo
        )

        opt_zc = data.y[:, utils.zrdim :].double()
        opt_qij, opt_pg, opt_qg = utils.decompose_vars_zc_JA(opt_zc)

        assert torch.allclose(opt_qij, opt_cqij, rtol=0, atol=1e-4), "qij not equal"
        assert torch.allclose(opt_qg, opt_cqg, rtol=0, atol=1e-4), "qg not equal"
        assert torch.allclose(opt_pg, opt_cpg, rtol=0, atol=1e-4), "pg not equal"

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

        p_switch = MLP_out[:, -utils.numSwitches :]
        n_switch_per_batch = torch.full((200, 1), utils.numSwitches).squeeze()

        if warm_start:
            topology = self.PhyR(p_switch.flatten(), n_switch_per_batch)
        else:
            topology = p_switch.flatten()

        graph_topo = torch.ones((200, utils.M), device=self.device)
        graph_topo[:, -utils.numSwitches :] = topology.view((200, -1))

        v = MLP_out[:, utils.M : utils.M + utils.N]
        v_phys = utils.vLow * (1 - v) + utils.vUpp * v
        v[:, 0] = 1

        p_flow = MLP_out[:, : utils.M]
        p_flow_phys = p_flow - 0.5

        pg, qg, p_flow_corrected, q_flow_corrected = utils.complete_JA(
            data.x,
            v_phys,
            p_flow_phys,
            graph_topo,
        )

        z = torch.cat((p_flow_corrected, v, graph_topo), dim=1)
        zc = torch.cat((q_flow_corrected, pg, qg), dim=1)
        return z, zc


class simple_MLP(nn.Module):
    def __init__(self, args, utils):
        super().__init__()
        output_dim = utils.M + utils.N + utils.numSwitches
        self.MLP = MLP(args, 2*utils.N, 4*utils.N, output_dim)
        self.device = args["device"]
        self.PhyR = getattr(utils, args["PhyR"])
    def forward(self, data, utils, warm_start=False):      
        n_switches = torch.sum(torch.sum(data.S, dim=1), dim=1) // 2

        x_flatten = data.x.view(200, -1)
        z = self.MLP(x_flatten)
        p_flow, v, topo = utils.decompose_vars_z_JA(z)
        if warm_start:
            topology = self.PhyR(topo.flatten().sigmoid(), n_switches)
        graph_topo = torch.ones((200, utils.M), device=self.device)
        graph_topo[:, -utils.numSwitches :] = topology.view(200, -1)
        v[:, 0] = 1
        pg, qg, p_flow_corrected, q_flow_corrected = utils.complete_JA(
            data.x_data, v, p_flow, graph_topo
        )
        z = torch.cat((p_flow_corrected, v, graph_topo), dim=1)
        zc = torch.cat((q_flow_corrected, pg, qg), dim=1)
        return z, zc