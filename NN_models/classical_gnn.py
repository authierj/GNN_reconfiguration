# global
import torch.nn as nn
from NN_layers.readout import *
from NN_layers.graph_NN import GCN, GNN

import torch.autograd.profiler as profiler

# local
from utils_JA import xgraph_xflatten, Modified_Sigmoid
import utils_JA


class GCN_Global_MLP_reduced_model(nn.Module):
    def __init__(self, args, utils):
        super().__init__()
        output_dim = utils.M + utils.N + utils.numSwitches
        self.GNN = GCN(args)
        self.readout = GlobalMLP_reduced(args, utils.N, output_dim)
        self.device = args["device"]
        self.PhyR = getattr(utils, args["PhyR"])
        if args["switchActivation"] == "sig":
            self.switch_activation = nn.Sigmoid()
        elif args["switchActivation"] == "mod_sig":
            self.switch_activation = Modified_Sigmoid()
        else:
            self.switch_activation = nn.Identity()

    def forward(self, data, utils, warm_start=False):
        # input of Rabab's NN
        # x_input = xgraph_xflatten(x, 200, first_node=True)

        x_input = data.x.view(200, -1, 2)
        mask = torch.arange(utils.M).unsqueeze(0) >= utils.M - data.numSwitches

        xg = self.GNN(data.x, data.edge_index)

        # x_nn = xgraph_xflatten(xg, 200, first_node=True).to(device=self.device)
        x_nn = xg.view(200, -1)
        out = self.readout(x_nn)  # [pij, v, p_switch]

        p_switch = self.switch_activation(out[:, -utils.numSwitches : :])
        graph_topo = torch.ones((x_nn.shape[0], utils.M), device=self.device)
        graph_topo[mask] = p_switch

        if warm_start:
            graph_topo = self.PhyR(graph_topo, data.numSwitches)

        v = out[:, utils.M : utils.M + utils.N]
        v[:, 0] = 1
        pg, qg, p_flow_corrected, q_flow_corrected = utils.complete_JA(
            x_input,
            v,
            out[:, 0 : utils.M],
            graph_topo,
            utils.A,
        )
        z = torch.cat((p_flow_corrected, v, graph_topo), dim=1)
        zc = torch.cat((q_flow_corrected, pg, qg), dim=1)
        return z, zc


class GCN_local_MLP(nn.Module):
    def __init__(self, args, utils):
        super().__init__()
        self.GNN = GCN(args)
        self.SMLP = SMLP(
            3 * args["hiddenFeatures"], 4 * args["hiddenFeatures"], args["dropout"]
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
        # input of Rabab's NN
        # x_input = xgraph_xflatten(x, 200, first_node=True)
        x_input = data.x.view(200, -1, 2)
        mask = torch.arange(utils.M, device=utils.device).unsqueeze(
            0
        ) >= utils.M - data.numSwitches.unsqueeze(1)

        xg = self.GNN(data.x, data.edge_index)  # B*N x F
        num_features = xg.shape[1]

        x_nn = xg.view(200, -1, num_features)  # B x N x F

        edges = data.edge_index[0, :].view(200, -1)
        parent_edges = edges[:, : edges.shape[1] // 2]
        child_edges = edges[:, edges.shape[1] // 2 :]

        parent_nodes = parent_edges[~mask]
        parent_switches = parent_edges[mask]
        child_nodes = child_edges[~mask]
        child_switches = child_edges[mask]

        x1 = xg[parent_switches, :]
        x2 = xg[child_switches, :]
        x_g = torch.sum(x_nn, dim=1)  # B x F
        x_g_extended = x_g.repeat_interleave(data.numSwitches, dim=0)

        SMLP_input = torch.cat((x1, x2, x_g_extended), dim=1)  # num_switches*B x 3F
        SMLP_out = self.SMLP(
            SMLP_input
        )  # num_switches*B x 4, [switch_prob, P_flow, V_parent, V_child]

        p_switch = self.switch_activation(SMLP_out[:, 0])
        graph_topo = torch.ones((x_input.shape[0], utils.M), device=self.device)
        graph_topo[mask] = p_switch

        if warm_start:
            graph_topo = self.PhyR(graph_topo)

        ps_flow = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        ps_flow[mask] = SMLP_out[:, 1]
        vs_parent = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        vs_parent[mask] = SMLP_out[:, 2]
        vs_child = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        vs_child[mask] = SMLP_out[:, 3]

        xbegin = xg[parent_nodes, :]
        xend = xg[child_nodes, :]
        x_g_extended = x_g.repeat_interleave(utils.M - data.numSwitches, dim=0)

        CMLP_input = torch.cat((xbegin, xend, x_g_extended), dim=1)
        CMLP_out = self.CMLP(
            CMLP_input
        )  # (M-num_switch)*B x 3, [Pflow, Vparent, Vchild]

        pc_flow = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        pc_flow[~mask] = CMLP_out[:, 0]
        vc_parent = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        vc_parent[~mask] = CMLP_out[:, 1]
        vc_child = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        vc_child[~mask] = CMLP_out[:, 2]

        p_flow = ps_flow + pc_flow
        # first_mul = data.D_inv @ data.Incidence_parent
        # second_mul = first_mul @ Vc_parent.unsqueeze(2).double()
        B, N = data.inv_degree.shape
        D_inv = torch.zeros((B, N, N), device=self.device)
        D_inv[:, torch.arange(N, device=self.device), torch.arange(N, device=self.device)] = data.inv_degree

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
            x_input, v, p_flow, graph_topo, utils.A
        )

        z = torch.cat((p_flow_corrected, v, graph_topo), dim=1)
        zc = torch.cat((q_flow_corrected, pg, qg), dim=1)

        return z, zc


class GNN_global_MLP(nn.Module):
    def __init__(self, args, utils):
        super().__init__()
        output_dim = utils.M + utils.N + utils.numSwitches
        self.GNN = GNN(args)
        self.readout = GlobalMLP_reduced(args, utils.N, output_dim)
        self.device = args["device"]
        self.PhyR = getattr(utils, args["PhyR"])
        if args["switchActivation"] == "sig":
            self.switch_activation = nn.Sigmoid()
        elif args["switchActivation"] == "mod_sig":
            self.switch_activation = Modified_Sigmoid()
        else:
            self.switch_activation = nn.Identity()

    def forward(self, data, utils, warm_start=False):
        # input of Rabab's NN
        # x_input = xgraph_xflatten(x, 200, first_node=True)

        x_input = data.x.view(200, -1, 2)
        mask = torch.arange(utils.M).unsqueeze(0) >= utils.M - data.numSwitches

        xg = self.GNN(data.x, data.edge_index)

        x_nn = xgraph_xflatten(xg, 200, first_node=True).to(device=self.device)
        # x_nn = xg.view(200, -1, xg.shape[1])
        out = self.readout(x_nn)  # [pij, v, p_switch]

        p_switch = self.switch_activation(out[:, -utils.numSwitches : :])
        graph_topo = torch.ones((x_nn.shape[0], utils.M), device=self.device)
        graph_topo[mask] = p_switch

        if warm_start:
            graph_topo = self.PhyR(graph_topo, data.numSwitches)

        v = out[:, utils.M : utils.M + utils.N]
        v[:, 0] = 1
        pg, qg, p_flow_corrected, q_flow_corrected = utils.complete_JA(
            x_input,
            v,
            out[:, 0 : utils.M],
            graph_topo,
            utils.A,
        )

        z = torch.cat((p_flow_corrected, v, graph_topo), dim=1)
        zc = torch.cat((q_flow_corrected, pg, qg), dim=1)
        return z, zc


class GNN_local_MLP(nn.Module):
    def __init__(self, args, utils):
        super().__init__()
        self.GNN = GNN(args)
        self.SMLP = SMLP(
            3 * args["hiddenFeatures"], 4 * args["hiddenFeatures"], args["dropout"]
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
        # input of Rabab's NN
        # x_input = xgraph_xflatten(x, 200, first_node=True)
        x_input = data.x.view(200, -1, 2)
        mask = torch.arange(utils.M, device=utils.device).unsqueeze(
            0
        ) >= utils.M - data.numSwitches.unsqueeze(1)

        xg = self.GNN(data.x, data.edge_index)  # B*N x F
        num_features = xg.shape[1]

        x_nn = xg.view(200, -1, num_features)  # B x N x F

        edges = data.edge_index[0, :].view(200, -1)
        parent_edges = edges[:, : edges.shape[1] // 2]
        child_edges = edges[:, edges.shape[1] // 2 :]

        parent_nodes = parent_edges[~mask]
        parent_switches = parent_edges[mask]
        child_nodes = child_edges[~mask]
        child_switches = child_edges[mask]

        x1 = xg[parent_switches, :]
        x2 = xg[child_switches, :]
        x_g = torch.sum(x_nn, dim=1)  # B x F
        x_g_extended = x_g.repeat_interleave(data.numSwitches, dim=0)

        SMLP_input = torch.cat((x1, x2, x_g_extended), dim=1)  # num_switches*B x 3F
        SMLP_out = self.SMLP(
            SMLP_input
        )  # num_switches*B x 4, [switch_prob, P_flow, V_parent, V_child]

        p_switch = self.switch_activation(SMLP_out[:, 0])
        graph_topo = torch.ones((x_input.shape[0], utils.M), device=self.device)
        graph_topo[mask] = p_switch

        if warm_start:
            graph_topo = self.PhyR(graph_topo)

        ps_flow = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        ps_flow[mask] = SMLP_out[:, 1]
        vs_parent = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        vs_parent[mask] = SMLP_out[:, 2]
        vs_child = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        vs_child[mask] = SMLP_out[:, 3]

        xbegin = xg[parent_nodes, :]
        xend = xg[child_nodes, :]
        x_g_extended = x_g.repeat_interleave(utils.M - data.numSwitches, dim=0)

        CMLP_input = torch.cat((xbegin, xend, x_g_extended), dim=1)
        CMLP_out = self.CMLP(
            CMLP_input
        )  # (M-num_switch)*B x 3, [Pflow, Vparent, Vchild]

        pc_flow = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        pc_flow[~mask] = CMLP_out[:, 0]
        vc_parent = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        vc_parent[~mask] = CMLP_out[:, 1]
        vc_child = torch.zeros((x_nn.shape[0], utils.M), device=self.device)
        vc_child[~mask] = CMLP_out[:, 2]

        p_flow = ps_flow + pc_flow
        # first_mul = data.D_inv @ data.Incidence_parent
        # second_mul = first_mul @ Vc_parent.unsqueeze(2).double()
        # TODO put D_inv in the data object
        v = (
            utils.D_inv.float()
            @ data.inc_parents.float()
            @ (vc_parent + vs_parent).unsqueeze(2).float()
            + utils.D_inv.float()
            @ data.inc_childs.float()
            @ (vc_child + vs_child).unsqueeze(2).float()
        ).squeeze()
        v[:, 0] = 1  # V_PCC = 1

        pg, qg, p_flow_corrected, q_flow_corrected = utils.complete_JA(
            x_input, v, p_flow, graph_topo, utils.A
        )

        z = torch.cat((p_flow_corrected, v, graph_topo), dim=1)
        zc = torch.cat((q_flow_corrected, pg, qg), dim=1)

        return z, zc


class GCN_Global_MLP(nn.Module):
    def __init__(self, args, utils):
        super().__init__()
        output_dim = utils.M + utils.N + utils.numSwitches
        self.GNN = GCN(args)
        self.readout = GlobalMLP(args, utils.N, output_dim)

    def forward(self, data, utils):
        # input of Rabab's NN
        x_input = xgraph_xflatten(data.x, 200, first_node=False)

        xg = self.GNN(data.x, data.edge_index)
        x_nn = xgraph_xflatten(xg, 200, first_node=True)

        out = self.readout(x_nn)
        z = utils.output_layer(out)

        z, zc = utils.complete(x_input, z)
        zc_tensor = torch.stack(list(zc), dim=0)
        return z, zc_tensor, x_input
