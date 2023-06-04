import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function


class Utils:
    def __init__(self, data, device):
        # network data
        self.A = data.A.to(device)  # incidence matrix
        self.M = data.M
        self.N = data.N
        self.numSwitches = data.numSwitches
        self.pl = data.pl
        self.ql = data.ql
        self.Rall = data.Rall.to(device)
        self.Xall = data.Xall.to(device)
        # self.pgUpp = data.pgUpp.to(device)  # move to data
        # self.pgLow = data.pgLow.to(device)  # basically always zero
        self.pgLow = 0
        # self.qgUpp = data.qgUpp.to(device)  # move to data
        # self.qgLow = data.qgLow.to(device)  # basically always zero
        self.qgLow = 0 
        self.vUpp = data.vUpp.to(device)
        self.vLow = data.vLow.to(device)
        self.S = data.S
        self.Adj = data.Adj
        self.D_inv = data.D_inv.to(device)
        self.Incidence_parent = data.mStart.to_dense().to(device)
        self.Incidence_child = data.mEnd.to_dense().to(device)
        self.zrdim = data.zdim
        self.device = device

    def decompose_vars_z_JA(self, z):
        """
        decompose_vars_z_JA returns the decomposition of the neural network guess of Jules' reduced model

        args:
            z: the neural network guess
                z = [pij, v, topolgy]
        return:
            pij: the active power flow through each line
            v: the voltage at each node
            topology: the topology of the network
        """

        pij = z[:, 0 : self.M]
        v = z[:, self.M : self.M + self.N]
        topology = z[:, self.M + self.N : :]

        return pij, v, topology

    def decompose_vars_zc_JA(self, zc):
        """
        decompose_vars_zc_JA returns the decomposition of the completion variables of Jules' reduced model
        args:
            zc: the completion variables
                zc = [qij, pg, qg]
        return:
            qij: the reactive power flow through each line
            pg: the active power generation at each node
            qg: the reactive power generation at each node
        """

        qij = zc[:, 0 : self.M]
        pg = zc[:, self.M : self.M + self.N]
        qg = zc[:, self.M + self.N : :]

        return qij, pg, qg

    def obj_fnc_JA(self, z, zc):
        """
        obj_fnc approximates the power loss via line losses using Rij * (Pij^2 + Qij^2)

        return: the approximate line power losses
        """

        pij, _, _ = self.decompose_vars_z_JA(z)
        qij, _, _ = self.decompose_vars_zc_JA(zc)

        fncval = torch.sum((pij**2 + qij**2) * self.Rall, dim=1)
        return fncval

    def PhyR(self, s, n_switches):
        """
        args:
            s: Input switch  probabilities prediction of the NN in a flattened vector
            n_switches: The number of switches in each batch
        return:
            The topology of the graph
        """
        n_switches = n_switches[0]
        L = (self.N - 1) - (self.M - n_switches).int()
        p_switch = s.view(200, -1)

        # Find the L-th largest values along each row
        _, indices = torch.topk(p_switch, L, dim=1, largest=True, sorted=True)

        # Create a mask of the same shape as p_switch
        mask = torch.zeros_like(p_switch, requires_grad=True, device=self.device)

        # Set the L largest values in each row to one
        topology = torch.scatter(mask, 1, indices, 1)

        return topology.flatten()

    def back_PhyR(self, s, n_switches):
        """
        args:
            s: Input switch  probabilities prediction of the NN in a flattened vector
            n_switches: The number of switches in each batch
        return:
            The topology of the graph
        """
        n_switches = n_switches[0]
        L = (self.N - 1) - (self.M - n_switches).int()
        p_switch = s.view(200, -1)

        # Find the L-th largest values along each row
        _, indices = torch.topk(p_switch, L, dim=1, largest=True, sorted=True)

        # Create a mask of the same shape as p_switch
        mask = torch.zeros_like(p, requires_grad=True, device=self.device)

        # Set the L largest values in each row to one
        topology = torch.scatter(mask, 1, indices, 1)
        result = p_switch + (topology - p_switch).detach()

        return result.flatten()

    def mod_PhyR(self, s, n_switches):
        """
        args:
            s: Input switch  probabilities prediction of the NN in a flattened vector
            n_switches: The number of switches in each batch
        return:
            The topology of the graph
        """
        n_switches = n_switches[0]
        L = (self.N - 1) - (self.M - n_switches).int()
        p_switch = s.view(200, -1)
        p = p_switch[:, :-1]

        # Find the L-th largest values along each row
        _, indices = torch.topk(p, L + 1, dim=1, largest=True, sorted=True)

        # Create a mask of the same shape as p_switch
        mask = torch.zeros_like(p, requires_grad=True, device=self.device)

        # Set the L largest values in each row to one
        ind = torch.stack(
            [torch.arange(p.size(0), device=self.device), indices[:, -1]], dim=1
        )

        topology = torch.scatter(mask, 1, indices[:, :-1], 1)
        topology[ind[:, 0], ind[:, 1]] = p[ind[:, 0], ind[:, 1]]
        topo = torch.hstack((topology, -p[ind[:, 0], ind[:, 1]].unsqueeze(1)))

        return topo.flatten()

    def back_mod_PhyR(self, s, n_switches):
        """
        args:
            s: Input switch  probabilities prediction of the NN in a flattened vector
            n_switches: The number of switches in each batch
        return:
            The topology of the graph
        """
        n_switches = n_switches[0]
        L = (self.N - 1) - (self.M - n_switches).int()
        p_switch = s.view(200, -1)
        p = p_switch[:, :-1]

        # Find the L-th largest values along each row
        _, indices = torch.topk(p, L + 1, dim=1, largest=True, sorted=True)

        # Create a mask of the same shape as p_switch
        mask = torch.zeros_like(p, requires_grad=True, device=self.device)

        # Set the L largest values in each row to one
        ind = torch.stack(
            [torch.arange(p.size(0), device=self.device), indices[:, -1]], dim=1
        )

        topology = torch.scatter(mask, 1, indices[:, :-1], 1)
        topology[ind[:, 0], ind[:, 1]] = p[ind[:, 0], ind[:, 1]]
        topo = torch.hstack((topology, -p[ind[:, 0], ind[:, 1]].unsqueeze(1)))
        result = p_switch + (topo - p_switch).detach()

        return result.flatten()

    def complete_JA(self, x, v, p_flow, topo):
        """
        return the completion variables to satisfy the power flow equations

        args:
            x: the input to the neural network
            v: the voltage magnitudes
            p_flow: the active power flows
            topo: the topology of the graph

        return:
            pg: the active power generation
            qg: the reactive power generation
            p_flow_corrected: the active power flows corrected for the topology
            q_flow_corrected: the reactive power flows corrected for the topology
        """
        # test = (v.unsqueeze(1) @ self.A.float()).squeeze()
        # valid = v[0,:] @ self.A.float()
        # assert torch.max(torch.abs(test[0,:] - valid)) < 1e-4


        # test1 = 0.5 * (v.unsqueeze(1).double() @ self.A).squeeze()
        # test2 = self.Rall * p_flow.double()
        # result = (test1 - test2)/ self.Xall

        q_flow = (
            0.5 * (v.unsqueeze(1).double() @ self.A).squeeze() - self.Rall * p_flow.double()
        ) / self.Xall

        # assert that the equation is satisfied with vi-vj == qij + pij
        # delta_v = incidence.T.float() @ v[0, :]
        # loss = 2 * (self.Rall * p_flow + self.Xall * q_flow)
        # diff = torch.square(delta_v - loss[0,:])
        # assert torch.max(diff) < 1e-5

        q_flow_corrected = topo * q_flow.double()
        p_flow_corrected = topo * p_flow

        pl = x[:, :, 0]
        ql = x[:, :, 1]

        # A = self.A.unsqueeze(0).expand(200, -1, -1)

        # test = torch.matmul(A.float(), p_flow_corrected.unsqueeze(-1)).squeeze()
        pg = pl + (self.A @ p_flow_corrected.unsqueeze(-1).double()).squeeze()
        qg = ql + (self.A @ q_flow_corrected.unsqueeze(-1).double()).squeeze()

        #debug:
        # pl0 = pl[0,:]
        # ql0 = ql[0,:]
        # p_flow_corrected0 = p_flow_corrected[0,:]   
        # q_flow_corrected0 = q_flow_corrected[0,:]
        # pg0 = pl0 - self.A.float() @ p_flow_corrected0
        # qg0 = ql0 - self.A.float() @ q_flow_corrected0

        # assert torch.max(torch.abs(pg0 - pg[0,:])) < 1e-4 and torch.max(torch.abs(qg0 - qg[0,:])) < 1e-4


        return pg, qg, p_flow_corrected, q_flow_corrected

    def ineq_resid_JA(self, z, zc, pg_upp, qg_upp, incidence):
        """
        ineq_resid returns the violation of the inequality constraints

        args:
            z: the output of the neural network
            zc: the completion variables
            idx: the index of the case
            incidence: the incidence matrix of the network
        return:
            violated_resids: the value of the violation of the inequality constraints
        """
        _, v, topology = self.decompose_vars_z_JA(z)
        _, pg, qg = self.decompose_vars_zc_JA(zc)

        pg_upp_resid = pg - pg_upp
        pg_low_resid = self.pgLow - pg

        qg_upp_resid = qg - qg_upp
        qg_low_resid = self.qgLow - qg

        v_upp_resid = v - self.vUpp
        v_low_resid = self.vLow - v

        # TODO discuss with Rabab
        connectivity = (
            1 - (torch.abs(incidence) @ topology.unsqueeze(2).double()).squeeze()
        )

        resids = torch.cat(
            [
                100*pg_upp_resid,
                100*pg_low_resid,
                100*qg_upp_resid,
                100*qg_low_resid,
                v_upp_resid,
                v_low_resid,
                connectivity,
                -topology,
            ],
            dim=1,
        )

        violated_resid = torch.clamp(resids, min=0)
        # print(violated_resid[:,0])
        idx = torch.argmax(violated_resid, dim=1)
        return violated_resid

    def prob_push(self, z):
        """
        prob_push returns a cost that pushes the switches towards zero or one probabilities and pushes the right number of switches to be closed the number of switches

        args:
            z: the output of the neural network
        return:
            push: the cost that pushes the switches towards zero or one probabilities and pushes the right number of switches to be closed the number of switches
        """

        _, _, topology = self.decompose_vars_z_JA(z)

        push_speed = torch.sum(-topology * (topology - 1), dim=1)
        physic_informed = (torch.sum(topology, dim=1) - (self.N - 1)) ** 2
        push = push_speed + physic_informed
        return push

    def cross_entropy_loss_topology(self, z, y):
        """
        cross_entropy_loss_topology returns the cross entropy loss between the topology chosen by the neural network and the reference topology

        args:
            z_hat: the output of the neural network
            y: the reference solution
            switch_mask: the mask for the switches

        return:
            loss: the cross entropy loss between the topology chosen by the neural network and the reference topology
        """

        _, _, opt_topo = self.decompose_vars_z_JA(y[:, : self.zrdim])
        _, _, topology = self.decompose_vars_z_JA(z)

        criterion = nn.BCELoss(reduction="sum")
        return criterion(topology, opt_topo)

    def squared_error_topology(self, z, y):
        """
        squared_error_topology returns the squared error between the topology chosen by the neural network and the reference topology

        args:
            z_hat: the output of the neural network
            y: the reference solution
            switch_mask: the mask for the switches

        return:
            loss: the squared error between the topology chosen by the neural network and the reference topology
        """
        _, _, opt_topo = self.decompose_vars_z_JA(y[:, : self.zrdim])
        _, _, topology = self.decompose_vars_z_JA(z)

        criterion = nn.MSELoss(reduction="none")
        distance = criterion(topology, opt_topo)
        return torch.sum(distance, dim=1)

    def opt_topology_dist_JA(self, z, y):
        """
        opt_dispatch_dist_JA returns the squared distance between the topology chosen by the neural network and the reference topology

        args:
            z_hat: the output of the neural network
            y: the reference solution
            switch_mask: the mask for the switches

        return:
            delta_topo: the squared distance between the topology chosen by the neural network and the reference topology
        """

        _, _, opt_topo = self.decompose_vars_z_JA(y[:, : self.zrdim])
        _, _, topology = self.decompose_vars_z_JA(z)

        delta_topo = torch.square(topology - opt_topo)

        sum = torch.sum(delta_topo, dim=1)
        max = torch.max(torch.sum(delta_topo, dim=1))
        # if torch.max(torch.sum(delta_topo, dim=1)) > 4:
        #     idx = torch.argmax(torch.sum(delta_topo, dim=1))
        #     print("problem")

        return delta_topo

    def opt_dispatch_dist_JA(self, zc, y):
        """
        opt_dispatch_dist_JA returns the squared distance between the output of
        the neural network and the completion variable from the reference solution

        args:
            zc: the completion variables
            y: the reference solution
        return:
            dispatch_resid: the squared distance between the output of the neural network and the completion variable from the reference solution
        """

        _, opt_pg, opt_qg = self.decompose_vars_zc_JA(y[:, self.zrdim : :])
        _, pg, qg = self.decompose_vars_zc_JA(zc)

        delta_pg = torch.square(pg - opt_pg)
        delta_qg = torch.square(qg - opt_qg)

        dist = torch.cat([delta_pg, delta_qg], dim=1)
        return dist

    def opt_voltage_dist_JA(self, z, y):
        """
        opt_voltage_dist_JA returns the squared distance between the output of
        the neural network and the completion variable from the reference solution

        args:
            z: the output of the neural network
            y: the reference solution
        return:
            delta_v: the squared distance between the voltage predicted by the neural network and the reference voltage
        """

        _, opt_v, _ = self.decompose_vars_z_JA(y[:, : self.zrdim])
        _, v, _ = self.decompose_vars_z_JA(z)

        delta_v = torch.square(v[:, 1:] - opt_v[:, 1:])

        return delta_v

    def opt_gap_JA(self, z, zc, y):
        """
        opt_gap_JA returns the optimality gap of the neural network guess

        args:
            z: the output of the neural network
            zc: the completion variables
            y: the reference solution
        return:
            opt_gap: the optimality gap of the neural network guess
        """
        opt_z = y[:, : self.zrdim]
        opt_zc = y[:, self.zrdim :]

        opt_cost = self.obj_fnc_JA(opt_z, opt_zc)
        cost = self.obj_fnc_JA(z, zc)
        ineq_dist = self.ineq_resid_JA(z, zc, pg_upp, qg_upp, incidence) 
        ineq_cost = torch.linalg.vector_norm(ineq_dist, dim=1)  # gives norm for scalar weight

        soft_weight = args["softWeight"]

        total_loss = obj_cost + soft_weight * ineq_cost
        return total_loss



        opt_gap = (cost - opt_cost) / opt_cost
        return opt_gap

    def average_sum_distance(self, z_hat, zc_hat, y, switch_mask, zrdim):
        """
        average_sum_distance returns the average over a batch of the sum of the distances between the variables and the reference solution

        args:
            z_hat: the output of the neural network
            zc_hat: the completion variables
            y: the reference solution
            switch_mask: the mask for the switches
            zrdim: the dimension of the output of rabab's network
        return:
            average_sum_distance: the average over a batch of the sum of the distances between the variables and the reference solution
        """

        dispatch_resid = self.dist_opt_dispatch(
            z_hat, zc_hat, y, switch_mask, zrdim
        )  # B x 3M + 3N [pij,qij, y, v, pg, qg]
        batch_mean_resid = torch.mean(dispatch_resid, dim=0)
        average_sum_distance = torch.zeros(6)
        average_sum_distance[0] = torch.sum(batch_mean_resid[0 : self.M])
        average_sum_distance[1] = torch.sum(batch_mean_resid[self.M : 2 * self.M])
        average_sum_distance[2] = torch.sum(batch_mean_resid[2 * self.M : 3 * self.M])
        average_sum_distance[3] = torch.sum(
            batch_mean_resid[3 * self.M : 3 * self.M + self.N]
        )
        average_sum_distance[4] = torch.sum(
            batch_mean_resid[3 * self.M + self.N : 3 * self.M + 2 * self.N]
        )
        average_sum_distance[5] = torch.sum(
            batch_mean_resid[3 * self.M + 2 * self.N : 3 * self.M + 3 * self.N]
        )

        return average_sum_distance

    def optimality_distance(self, z_hat, zc_hat, y):
        """
        optimality distance returns the distance between the guess of the neural network and the reference solution
        """
        opt_z = y[:, : self.zrdim]
        opt_zc = y[:, self.zrdim :]

        pij, v, topology = self.decompose_vars_z_JA(z_hat)
        qij, pg, qg = self.decompose_vars_zc_JA(zc_hat)

        opt_pij, opt_v, opt_topology = self.decompose_vars_z_JA(opt_z)
        opt_qij, opt_pg, opt_qg = self.decompose_vars_zc_JA(opt_zc)

        pij_gap = torch.sum(torch.square(pij - opt_pij), dim=1)
        qij_gap = torch.sum(torch.square(qij - opt_qij), dim=1)
        v_gap = torch.sum(torch.square(v - opt_v), dim=1)
        pg_gap = torch.sum(torch.square(pg - opt_pg), dim=1)
        qg_gap = torch.sum(torch.square(qg - opt_qg), dim=1)
        topo_gap = torch.sum(torch.square(topology - opt_topology), dim=1)

        return pij_gap + v_gap + topo_gap


def xgraph_xflatten(x_graph, batch_size, first_node=False):
    # TODO see how the epochs
    """
    xgraph_xflatten returns the input of the GNN as expected by the NN

    :param x_graph: the input of the GNN
    :param num_features: the number of features per node
    :param batch_size: the size of the batches
    :param first_node: determine if the first node must be kept or not

    :return: the input of the GNN as expected by the NN
    """
    graph_3d = torch.reshape(
        x_graph, (batch_size, int(x_graph.shape[0] / batch_size), x_graph.shape[1])
    )

    if not first_node:
        graph_3d = graph_3d[:, 1::, :]
    xNN_3d = torch.transpose(graph_3d, 1, 2)
    xNN = torch.flatten(xNN_3d, 1, 2)
    return xNN


def total_loss(z, zc, criterion, utils, args, pg_upp, qg_upp, incidence, y):
    """
    total loss returns the sum of the loss function and norm of the violation of
    the inequality constraints multiplied by the soft weight

    args:
        z: the output of the neural network
        zc: the completion variables
        criterion: the loss function
        utils: the utils object
        args: the arguments given by the user
        idx: the index of the the training data
        incidence: the incidence matrix
        train: boolean to determine if we are in training or not
    return:
        total_loss: the total loss
        soft_weight: the soft weight
    """

    obj_cost = criterion(z, zc)
    supervised_cost = utils.optimality_distance(z, zc, y)
    ineq_dist = utils.ineq_resid_JA(
        z, zc, pg_upp, qg_upp, incidence
    )  # gives update for vector weight
    # print(ineq_dist[0, :])
    ineq_cost = torch.linalg.vector_norm(
        ineq_dist, dim=1
    )  # gives norm for scalar weight

    soft_weight = args["softWeight"]

    total_loss = obj_cost + soft_weight * ineq_cost
    mean_loss = torch.mean(total_loss)
    return total_loss

def dict_agg(stats, key, value, op):
    """
    dict_agg is a function that aggregates values in a dictionary.

    args:
        stats: a dictionary
        key: the key of the dictionary
        value: the value to be aggregated
        op: the operation to be performed, op in ['sum', 'concat', 'vstack']

    returns: None
    """
    if key in stats.keys():
        if op == "sum":
            stats[key] += value
        elif op == "concat":
            stats[key] = np.hstack((stats[key], value))
        elif op == "vstack":
            stats[key] = np.vstack((stats[key], value))
        else:
            raise NotImplementedError
    else:
        stats[key] = value


def default_args():
    defaults = {}

    defaults["model"] = "GCN_local_MLP"
    defaults["network"] = "baranwu33"
    defaults["epochs"] = 500
    defaults["batchSize"] = 200
    defaults["lr"] = 1e-3  # NN learning rate
    defaults["dropout"] = 0.1
    defaults["numLayers"] = 4
    defaults["inputFeatures"] = 2
    defaults["hiddenFeatures"] = 4
    defaults["softWeight"] = 100  # this is lambda_g in the paper
    defaults["saveModel"] = False
    defaults["saveAllStats"] = False
    defaults["resultsSaveFreq"] = 50
    defaults["topoLoss"] = False
    defaults["topoWeight"] = 100
    defaults["aggregation"] = "max"
    defaults["norm"] = "batch"
    defaults["corrEps"] = 1e-3
    defaults["switchActivation"] = "sig"
    defaults["warmStart"] = False
    defaults["PhyR"] = "mod_PhyR"
    defaults["pushProb"] = False
    defaults["pushWeight"] = 100

    return defaults


class Modified_Sigmoid(nn.Module):
    def __init__(self, tau=5):
        super(Modified_Sigmoid, self).__init__()
        self.tau = tau

    def forward(self, p):
        return torch.clamp(
            2 / (1 + torch.exp(-self.tau * p)) - 1, 0
        )  # modified sigmoid
