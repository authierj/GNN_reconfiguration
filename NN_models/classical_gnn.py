# global
import torch.nn as nn
from NN_layers.readout import *
from NN_layers.graph_NN import GCN, GNN

import torch.autograd.profiler as profiler

# local
from utils_JA import xgraph_xflatten


class GCN_Global_MLP_reduced_model(nn.Module):
    def __init__(self, args, N, output_dim):
        super().__init__()
        self.GNN = GCN(args)
        self.readout = GlobalMLP_reduced(args, N, output_dim)
        self.device = args["device"]

    def forward(self, data, utils):
        # input of Rabab's NN
        # x_input = xgraph_xflatten(x, 200, first_node=True)

        x_input = data.x.view(200, -1, 2)

        xg = self.GNN(data.x, data.edge_index)

        x_nn = xgraph_xflatten(xg, 200, first_node=True).to(device=self.device)
        # x_nn = xg.view(200, -1, xg.shape[1]) 
        out = self.readout(x_nn)  # [pij, v, p_switch]

        p_switch = out[:, -utils.numSwitches : :]
        n_switch_per_batch = torch.full((200, 1), utils.numSwitches).squeeze()

        # topology = utils.physic_informed_rounding(
        #         p_switch.flatten(), n_switch_per_batch
        # )
        topology = p_switch.flatten().sigmoid()
        graph_topo = torch.ones((200, utils.M), device=self.device).float()
        graph_topo[:, -utils.numSwitches : :] = topology.view((200, -1))

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
    def __init__(self, args, N, output_dim):
        super().__init__()
        self.GNN = GCN(args)
        self.SMLP = SMLP(
            3 * args["hiddenFeatures"], 4 * args["hiddenFeatures"], args["dropout"]
        )
        self.CMLP = CMLP(
            3 * args["hiddenFeatures"], 3 * args["hiddenFeatures"], args["dropout"]
        )
        self.completion_step = args["useCompl"]
        self.device = args["device"]

    def forward(self, data, utils):
        # input of Rabab's NN
        # x_input = xgraph_xflatten(x, 200, first_node=True)
        x_input = data.x.view(200, -1, 2)
        
        
        xg = self.GNN(data.x, data.edge_index)  # B*N x F
        num_features = xg.shape[1]

        x_nn = xg.view(200, -1, num_features)  # B x N x F

        switches_nodes = torch.nonzero(utils.S.triu())
        n_switch_per_batch = torch.full((200, 1), utils.numSwitches).squeeze()

        x_1 = x_nn[:, switches_nodes[:, 0], :].view(-1, num_features)
        x_2 = x_nn[:, switches_nodes[:, 1], :].view(-1, num_features)

        x_g = torch.sum(x_nn, dim=1)  # B x F
        x_g_extended = x_g.repeat(1, utils.numSwitches).view(-1, num_features)  # dim = num_switches*B x F

        SMLP_input = torch.cat((x_1, x_2, x_g_extended), dim=1)  # num_switches*B x 3F
        SMLP_out = self.SMLP(
            SMLP_input
        )  # num_switches*B x 4, [switch_prob, P_flow, V_parent, V_child]

        # topology = utils.physic_informed_rounding(
        #     SMLP_out[:, 0], n_switch_per_batch
        # )  # num_switches*B
        topology = SMLP_out[:, 0].sigmoid()
        graph_topo = torch.ones((200, utils.M), device=self.device).float()
        graph_topo[:, -utils.numSwitches : :] = topology.view((200, -1))

        ps_flow = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        ps_flow[:, -utils.numSwitches : :] = SMLP_out[:, 1].view((200, -1))
        vs_parent = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        vs_parent[:, -utils.numSwitches : :] = SMLP_out[:, 2].view((200, -1))
        vs_child = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        vs_child[:, -utils.numSwitches : :] = SMLP_out[:, 3].view((200, -1))

        nodes = torch.nonzero(utils.Adj.triu())  # dim = (M-num_switch)*B x 3

        x_begin = x_nn[:, nodes[:, 0], :].view(-1, num_features)
        x_end = x_nn[:, nodes[:, 1], :].view(-1, num_features)
        # x_g_extended = x_g[nodes[:, 0], :]
        x_g_extended = x_g.repeat(1, utils.M - utils.numSwitches).view(-1, num_features)

        CMLP_input = torch.cat((x_begin, x_end, x_g_extended), dim=1)
        CMLP_out = self.CMLP(
            CMLP_input
        )  # (M-num_switch)*B x 3, [Pflow, Vparent, Vchild]

        pc_flow = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        pc_flow[:, : -utils.numSwitches] = CMLP_out[:, 0].view((200, -1))
        vc_parent = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        vc_parent[:, : -utils.numSwitches] = CMLP_out[:, 1].view((200, -1))
        vc_child = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        vc_child[:, : -utils.numSwitches] = CMLP_out[:, 2].view((200, -1))

        p_flow = ps_flow + pc_flow
        # first_mul = data.D_inv @ data.Incidence_parent
        # second_mul = first_mul @ Vc_parent.unsqueeze(2).double()
        v = (
            utils.D_inv.float()
            @ utils.Incidence_parent.float()
            @ (vc_parent + vs_parent).unsqueeze(2).float()
            + utils.D_inv.float()
            @ utils.Incidence_child.float()
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
    def __init__(self, args, N, output_dim):
        super().__init__()
        self.GNN = GNN(args)
        self.readout = GlobalMLP_reduced(args, N, output_dim)
        self.device = args["device"]
        
    def forward(self, data, utils):
        # input of Rabab's NN
        x_input = data.x.view(200, -1, 2)

        xg = self.GNN(data.x, data.edge_index)

        x_nn = xgraph_xflatten(xg, 200, first_node=True).to(device=self.device)
        # x_nn = xg.view(200, -1, xg.shape[1]) 
        out = self.readout(x_nn)  # [pij, v, p_switch]

        p_switch = out[:, -utils.numSwitches : :]
        n_switch_per_batch = torch.full((200, 1), utils.numSwitches).squeeze()

        topology = utils.physic_informed_rounding(
                p_switch.flatten(), n_switch_per_batch
        )
        graph_topo = torch.ones((200, utils.M), device=self.device).bool()
        graph_topo[:, -utils.numSwitches : :] = topology.view((200, -1))

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
    def __init__(self, args, N, output_dim):
        super().__init__()
        self.GNN = GNN(args)
        self.SMLP = SMLP(
            3 * args["hiddenFeatures"], 4 * args["hiddenFeatures"], args["dropout"]
        )
        self.CMLP = CMLP(
            3 * args["hiddenFeatures"], 3 * args["hiddenFeatures"], args["dropout"]
        )
        self.completion_step = args["useCompl"]
        self.device = args["device"]
        
    def forward(self, data, utils):
        x_input = data.x.view(200, -1, 2)
        
        
        xg = self.GNN(data.x, data.edge_index)  # B*N x F
        num_features = xg.shape[1]

        x_nn = xg.view(200, -1, num_features)  # B x N x F

        switches_nodes = torch.nonzero(utils.S.triu())
        n_switch_per_batch = torch.full((200, 1), utils.numSwitches).squeeze()

        x_1 = x_nn[:, switches_nodes[:, 0], :].view(-1, num_features)
        x_2 = x_nn[:, switches_nodes[:, 1], :].view(-1, num_features)

        x_g = torch.sum(x_nn, dim=1)  # B x F
        x_g_extended = x_g.repeat(1, utils.numSwitches).view(-1, num_features)  # dim = num_switches*B x F

        SMLP_input = torch.cat((x_1, x_2, x_g_extended), dim=1)  # num_switches*B x 3F
        SMLP_out = self.SMLP(
            SMLP_input
        )  # num_switches*B x 4, [switch_prob, P_flow, V_parent, V_child]

        topology = utils.physic_informed_rounding(
            SMLP_out[:, 0], n_switch_per_batch
        )  # num_switches*B
        graph_topo = torch.ones((200, utils.M)).bool().to(self.device)
        graph_topo[:, -utils.numSwitches : :] = topology.view((200, -1))

        ps_flow = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        ps_flow[:, -utils.numSwitches : :] = SMLP_out[:, 1].view((200, -1))
        vs_parent = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        vs_parent[:, -utils.numSwitches : :] = SMLP_out[:, 2].view((200, -1))
        vs_child = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        vs_child[:, -utils.numSwitches : :] = SMLP_out[:, 3].view((200, -1))

        nodes = torch.nonzero(utils.Adj.triu())  # dim = (M-num_switch)*B x 3

        x_begin = x_nn[:, nodes[:, 0], :].view(-1, num_features)
        x_end = x_nn[:, nodes[:, 1], :].view(-1, num_features)
        # x_g_extended = x_g[nodes[:, 0], :]
        x_g_extended = x_g.repeat(1, utils.M - utils.numSwitches).view(-1, num_features)

        CMLP_input = torch.cat((x_begin, x_end, x_g_extended), dim=1)
        CMLP_out = self.CMLP(
            CMLP_input
        )  # (M-num_switch)*B x 3, [Pflow, Vparent, Vchild]

        pc_flow = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        pc_flow[:, : -utils.numSwitches] = CMLP_out[:, 0].view((200, -1))
        vc_parent = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        vc_parent[:, : -utils.numSwitches] = CMLP_out[:, 1].view((200, -1))
        vc_child = torch.zeros((x_nn.shape[0], utils.M)).to(self.device)
        vc_child[:, : -utils.numSwitches] = CMLP_out[:, 2].view((200, -1))

        p_flow = ps_flow + pc_flow
        # first_mul = data.D_inv @ data.Incidence_parent
        # second_mul = first_mul @ Vc_parent.unsqueeze(2).double()
        v = (
            utils.D_inv.float()
            @ utils.Incidence_parent.float()
            @ (vc_parent + vs_parent).unsqueeze(2).float()
            + utils.D_inv.float()
            @ utils.Incidence_child.float()
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
    def __init__(self, GNN, readout, completion_step):
        super().__init__()
        self.GNN = GNN
        self.readout = readout
        self.completion_step = completion_step

    def forward(self, data, utils):
        # input of Rabab's NN
        x_input = xgraph_xflatten(data.x, 200, first_node=False)

        xg = self.GNN(data.x, data.edge_index)
        x_nn = xgraph_xflatten(xg, 200, first_node=True)

        out = self.readout(x_nn)
        z = utils.output_layer(out)

        if self.completion_step:
            z, zc = utils.complete(x_input, z)
            zc_tensor = torch.stack(list(zc), dim=0)
            return z, zc_tensor, x_input

        else:
            return utils.process_output(x_input, out), x_input
