import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function


class Utils:
    def __init__(self, data, device):
        self.A = data.A.to(device)
        self.M = data.M
        self.N = data.N
        self.numSwitches = data.numSwitches
        self.pl = data.pl
        self.ql = data.ql
        self.Rall = data.Rall.to(device)
        self.Xall = data.Xall.to(device)
        self.pgUpp = data.pgUpp.to(device)
        self.pgLow = data.pgLow.to(device)
        self.qgUpp = data.qgUpp.to(device)
        self.qgLow = data.qgLow.to(device)
        self.bigM = data.bigM
        self.Xall = data.Xall.to(device)
        self.Rall = data.Rall.to(device)
        self.vUpp = data.vUpp.to(device)
        self.vLow = data.vLow.to(device)
        self.Aabs = data.Aabs
        self.dineq_dz_fullmat = data.dineq_dz_fullmat
        self.dzc_dz_mat = data.dzc_dz_mat
        self.dzc_dz = data.dzc_dz
        self.dzc_dx = data.dzc_dx
        self.S = data.S
        self.Adj = data.Adj
        self.D_inv = data.D_inv.to(device)
        self.Incidence_parent = data.mStart.to_dense().to(device)
        self.Incidence_child = data.mEnd.to_dense().to(device)
        self.zrdim = data.zdim
        self.device = device

    def decompose_vars_z(self, z):
        """
        decompose_vars returns the decomposition of the neural network guess

        :param z: the neural network guess
                    z = [zji, y\last, pij, pji, qji, qij_sw, v\f, plf, qlf]
        :return: the decomposition of z
        """

        zji = z[:, 0 : self.M]
        y_nol = z[:, self.M + np.arange(0, self.numSwitches - 1)]
        pij = z[:, self.M + self.numSwitches - 1 + np.arange(0, self.M)]
        pji = z[:, 2 * self.M + self.numSwitches - 1 + np.arange(0, self.M)]
        qji = z[:, 3 * self.M + self.numSwitches - 1 + np.arange(0, self.M)]
        qij_sw = z[
            :, 4 * self.M + self.numSwitches - 1 + np.arange(0, self.numSwitches)
        ]
        v = z[
            :, 4 * self.M + 2 * self.numSwitches - 1 + np.arange(0, self.N - 1)
        ]  # feeder not included
        plf = z[:, 4 * self.M + 2 * self.numSwitches - 1 + self.N - 1]
        qlf = z[:, 4 * self.M + 2 * self.numSwitches - 1 + self.N]

        return zji, y_nol, pij, pji, qji, qij_sw, v, plf, qlf

    def decompose_vars_zc(self, zc):
        # zc = [zij, ylast, {qij}nosw, pg, qg]  # completion vars
        zij = zc[:, np.arange(0, self.M)]
        ylast = zc[:, self.M]
        qij_nosw = zc[:, self.M + 1 + np.arange(0, self.M - self.numSwitches)]
        pg = zc[:, 2 * self.M + 1 - self.numSwitches + np.arange(0, self.N)]
        qg = zc[:, 2 * self.M + 1 - self.numSwitches + self.N + np.arange(0, self.N)]

        return zij, ylast, qij_nosw, pg, qg

    def get_integer_z(self, z):
        z_ij, y_nol, _, _, _, _, _, _, _ = self.decompose_vars_z(z)

        return torch.cat([z_ij, y_nol], dim=1)

    def obj_fnc(self, z, zc):
        """
        obj_fnc approximates the power loss via line losses using
        Rij * (Pij^2 + Pji^2 + Qij^2 + Qji^2)

        :param z: the output of the neural network
        :param zc: the completion variables as defined in the paper
        :return: the approximate line power losses
        """
        _, _, pij, pji, qji, qij_sw, _, _, _ = self.decompose_vars_z(z)
        _, _, qij_nosw, _, _ = self.decompose_vars_zc(zc)

        qij = torch.hstack((qij_nosw, qij_sw))

        fncval = torch.sum(
            torch.mul(
                torch.square(pij)
                + torch.square(pji)
                + torch.square(qij)
                + torch.square(qji),
                self.Rall,
            ),
            dim=1,
        )

        return fncval

    def eq_resid(self, z, zc):
        # should be zero, but implementing it as a debugging step
        # pl, ql = self.decompose_vars_x(x)
        zji, y_nol, pij, pji, qji, qij_sw, v, plf, qlf = self.decompose_vars_z(z)
        zij, ylast, qij_nosw, pg, qg = self.decompose_vars_zc(zc)

        qij = torch.hstack((qij_nosw, qij_sw))
        y = torch.hstack((y_nol, ylast.unsqueeze(1)))

        ncases = z.shape[0]
        numsw = self.numSwitches
        vall = torch.hstack((torch.ones(ncases, 1), v))

        all_ohm_eqns = torch.matmul(-vall, self.A.double()) + 2 * (
            torch.mul((pij - pji), self.Rall) + torch.mul((qij - qji), self.Xall)
        )

        # TODO split-up this equation
        resids = torch.cat(
            [
                zij[:, 0 : self.M - numsw]
                + zji[:, 0 : self.M - numsw]
                - torch.ones(ncases, self.M - numsw),
                zij[:, self.M - numsw :] + zji[:, self.M - numsw :] - y,
                (torch.sum(y, 1) - ((self.N - 1) - (self.M - numsw))).unsqueeze(1),
                # should this be pji - pij (equivalently reverse sign on the full term)
                pg
                - (
                    torch.hstack((plf.unsqueeze(1), self.pl)).T
                    + torch.mm(self.A.double(), torch.transpose(pij - pji, 0, 1))
                ).T,
                all_ohm_eqns[:, 0 : self.M - numsw],
                # should this be qji - qij (equivalently reverse sign on the full term)
                qg
                - (
                    torch.hstack((qlf.unsqueeze(1), self.ql)).T
                    + torch.mm(self.A.double(), torch.transpose(qij - qji, 0, 1))
                ).T,
            ],
            dim=1,
        )

        return resids

    def ineq_resid(self, z, zc, idx):  # Z = [z,zc]
        zji, y_nol, pij, pji, qji, qij_sw, v, plf, qlf = self.decompose_vars_z(z)
        zij, ylast, qij_nosw, pg, qg = self.decompose_vars_zc(zc)

        qij = torch.hstack((qij_nosw, qij_sw))
        y = torch.hstack((y_nol, ylast.unsqueeze(1)))

        ncases = z.shape[0]  # = batch size
        vall = torch.hstack((torch.ones(ncases, 1), v))

        pg_upp_resid = pg[:, 1:None] - self.pgUpp[idx, :]
        pg_low_resid = self.pgLow[idx, :] - pg[:, 1:None]

        qg_upp_resid = qg[:, 1:None] - self.qgUpp[idx, :]
        qg_low_resid = self.qgLow[idx, :] - qg[:, 1:None]

        v_upp_resid = v - self.vUpp
        v_low_resid = v - self.vLow

        # TODO rewrite these and find better names
        matrix1 = (
            torch.mm(
                torch.neg(
                    torch.transpose(
                        torch.index_select(
                            self.A,
                            1,
                            torch.from_numpy(
                                np.arange(self.M - self.numSwitches, self.M)
                            ).long(),
                        ),
                        0,
                        1,
                    )
                ),
                vall.T.double(),
            ).T
            + 2
            * (torch.mul((pij - pji), self.Rall) + torch.mul((qij - qji), self.Xall))[
                :, -self.numSwitches : None
            ]
            - self.bigM * (1 - y)
        )

        matrix2 = (
            -torch.mm(
                torch.neg(
                    torch.transpose(
                        torch.index_select(
                            self.A,
                            1,
                            torch.from_numpy(
                                np.arange(self.M - self.numSwitches, self.M)
                            ).long(),
                        ),
                        0,
                        1,
                    )
                ),
                vall.T.double(),
            ).T
            - 2
            * (torch.mul((pij - pji), self.Rall) + torch.mul((qij - qji), self.Xall))[
                :, -self.numSwitches : None
            ]
            - self.bigM * (1 - y)
        )

        # TODO rewrite that in multiple steps
        resids = torch.cat(
            [
                pg_upp_resid,
                pg_low_resid,
                qg_upp_resid,
                qg_low_resid,
                -torch.reshape(plf, (-1, 1)),
                -torch.reshape(qlf, (-1, 1)),
                -torch.reshape(pg[:, 0], (-1, 1)),
                -torch.reshape(qg[:, 0], (-1, 1)),
                v_upp_resid,
                v_low_resid,
                pij - self.bigM,
                -pij,
                pji - self.bigM,
                -pji,
                qij - self.bigM,
                -qij,
                qji - self.bigM,
                -qji,
                -zij,
                -zji,
                -y,
                matrix1,
                matrix2,
                1 - torch.mm(self.Aabs.double(), (zij + zji).T).T,
            ],
            dim=1,
        )

        return torch.clamp(resids, 0)

    def corr_steps(self, z, zc, idx):
        pos_resids = self.ineq_resid(z, zc, idx)
        delz = 2 * torch.mm(pos_resids, self.dineq_dz_fullmat)
        delzc = torch.matmul(self.dzc_dz_mat, delz.T).T
        return delz, delzc

    def corr_steps_partial(self, z, zc, idx):
        pos_resids = self.ineq_resid(z, zc, idx)
        delz = 2 * torch.mm(pos_resids, self.dineq_dz_partialmat)
        delzc = torch.matmul(self.dzc_dz_mat, delz.T).T
        return delz, delzc

    def output_layer(self, nnout):
        # apply mixed sigmoid to zji var
        # apply random {0,1} selection to {y}\last
        # map other vars to physics w/ standard sigmoid

        # parameter tau controls sharpness of the sigmoid function applied to binary variables zji. larger = sharper
        tau = 5

        # z = mixedIntFnc(self)(nnout, tau)
        z = self.mixedIntOutput(nnout, tau)

        # z = [zji, y\last, pij, pji, qji, {qij}sw, v\f, plf, qlf]  # initial guess from NN
        # zji and y\last are binary vars, which have already been processed to be between 0 and 1
        # process pij, pji, qji, {qij}_sw, v\f, plf, qlf to be within physical bounds
        (
            zji,
            y_nol,
            pij_nn,
            pji_nn,
            qji_nn,
            qij_nosw_nn,
            v_nn,
            plf,
            qlf,
        ) = self.decompose_vars_z(z)
        pij = (
            pij_nn * self.bigM
        )  # pij_min = 0; don't enforce any power flow constraints here
        pji = (
            pji_nn * self.bigM
        )  # pji_min = 0; don't enforce any power flow constraints here
        qji = (
            qji_nn * self.bigM
        )  # qji_min = 0; don't enforce any power flow constraints here
        qij_nosw = qij_nosw_nn * self.bigM
        v = v_nn * self.vUpp + (1 - v_nn) * self.vLow
        z_physical = torch.hstack(
            (zji, y_nol, pij, pji, qji, qij_nosw, v, plf.unsqueeze(1), qlf.unsqueeze(1))
        )

        return z_physical

    def mixedIntOutput(self, z, tau):
        bin_vars_zji = z[:, 0 : self.M]
        bin_vars_y = z[:, self.M : (self.M + self.numSwitches - 1)]
        cont_vars = z[:, self.M + self.numSwitches - 1 : None]
        # classical sigmoid for continuous variables
        output_cont = nn.Sigmoid()(cont_vars)

        # modified sigmoid
        output_bin_zji = torch.clamp(2 / (1 + torch.exp(-tau * bin_vars_zji)) - 1, 0)

        batchsize = z.size(dim=0)
        # r = np.random.randint(-1, 1, batchsize)

        # PHYSIC INFORMED ROUNDING

        L_min = (self.N - 1) - (self.M - self.numSwitches)
        y_sorted_inds = torch.argsort(bin_vars_y)  # sorted in ascending order

        # output_bin_y = bin_vars_y.abs()
        # output_bin_y = nn.Sigmoid()(bin_vars_y) # cast between 0 and 1
        output_bin_y = torch.clamp(bin_vars_y, 0, 1)
        # ceil the largest L values to 1, floor the smallest size(bin_y)-L values to 0
        rows_to_ceil = np.hstack(
            (np.arange(0, batchsize).repeat(L_min),)
        )  # np.where(r == 0)[0]
        cols_to_ceil = np.hstack(
            (y_sorted_inds[:, -L_min:].flatten(),)
        )  # y_sorted_inds[np.where(r == 0)[0], -L_min-1]

        num_to_zero = bin_vars_y.size(dim=1) - L_min - 1
        if num_to_zero < 0:
            num_to_zero = 0
        rows_to_floor = np.hstack(
            (np.arange(0, batchsize).repeat(num_to_zero),)
        )  # np.where(r == -1)[0]
        cols_to_floor = np.hstack(
            (y_sorted_inds[:, 0:num_to_zero].flatten(),)
        )  # y_sorted_inds[np.where(r == -1)[0], -L_min-1]

        # output_bin_y[rows_to_ceil, cols_to_ceil] = output_bin_y[rows_to_ceil, cols_to_ceil].ceil()
        # output_bin_y[rows_to_floor, cols_to_floor] = output_bin_y[rows_to_floor, cols_to_floor].floor()
        output_bin_y[rows_to_ceil, cols_to_ceil] = 1
        output_bin_y[rows_to_floor, cols_to_floor] = 0

        z_new = torch.hstack((output_bin_zji, output_bin_y, output_cont))
        return z_new


    def decompose_vars_x(self, x):
        # x = [pl\f, ql\f]  # input to NN
        pl = x[:, 0 : self.N - 1]
        ql = x[:, self.N - 1 : None]
        return pl, ql

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

    def physic_informed_rounding(self, s, n_switches):
        """
        args:
            s: Input switch  probabilities prediction of the NN in a flattened vector
            n_switches: The number of switches in each batch
        return:
            The topology of the graph
        """

        """
        switch_indices = torch.cumsum(n_switches, dim=0)
        begin = switch_indices[0:-1]
        end = switch_indices[1:]
        topology = torch.zeros_like(s).bool()
        L_min = (self.N - 1) - (self.M - n_switches).int()

        s_idx_before = 0
        i = 0

        for s_idx in switch_indices:
            closed_indices = torch.topk(s[s_idx_before:s_idx], k=L_min[i]).indices
            topology[s_idx_before:s_idx][closed_indices] = True
            i = i + 1
            s_idx_before = s_idx

        return topology

        n_switches = n_switches[0]
        L_min = (self.N - 1) - (self.M - n_switches).int()
        p_switches = s.view(200, -1)

        top_l_values, top_l_indices = torch.topk(p_switches, L_min, dim=1)

        # create a mask indicating which elements in p_switch are in the top L values
        # mask = p_switches >= top_l_values[:, -1].unsqueeze(1)

        # create a tensor of zeros with the same shape as p_switch
        top_l = torch.zeros_like(p_switches, dtype=torch.bool, device=self.device)

        # use the mask to set the top L values to True
        top_l[top_l_indices] = True

        topology = top_l.flatten()
        """
        n_switches = n_switches[0]
        L = (self.N - 1) - (self.M - n_switches).int()
        p_switch = s.view(200, -1)
        
        # Find the L-th largest values along each row
        _, indices = torch.topk(p_switch, L, dim=1, largest=True, sorted=False)
        
        # Create a mask of the same shape as p_switch
        mask = torch.zeros_like(p_switch)
        
        # Set the L largest values in each row to one
        topology = torch.scatter(mask, 1, indices, 1)
        result = p_switch + (topology - p_switch).detach()

    
        return result.flatten()

        # y_sorted_inds = torch.argsort(p_switch)  # sorted in ascending order
        # p_switch_copy = p_switch.clone()

        # # output_bin_y = torch.clamp(bin_vars_y, 0, 1)
        # # ceil the largest L values to 1, floor the smallest size(bin_y)-L values to 0
        # rows_to_ceil = np.hstack((np.arange(0, 200).repeat(L))) # np.where(r == 0)[0]
        # cols_to_ceil = np.hstack((y_sorted_inds[:, -L:].flatten())) # y_sorted_inds[np.where(r == 0)[0], -L_min-1]

        # num_to_zero = p_switch.size(dim=1) - L
        # if num_to_zero > 0:
        #     rows_to_floor = np.hstack((np.arange(0, 200).repeat(num_to_zero))) # np.where(r == -1)[0]
        #     cols_to_floor = np.hstack((y_sorted_inds[:, 0:num_to_zero].flatten())) # y_sorted_inds[np.where(r == -1)[0], -L_min-1]
        # else:
        #     rows_to_floor = []
        #     cols_to_floor = []
        # # output_bin_y[rows_to_ceil, cols_to_ceil] = output_bin_y[rows_to_ceil, cols_to_ceil].ceil()
        # # output_bin_y[rows_to_floor, cols_to_floor] = output_bin_y[rows_to_floor, cols_to_floor].floor()
        # p_switch_copy[rows_to_ceil, cols_to_ceil] = torch.maximum(p_switch[rows_to_ceil, cols_to_ceil], torch.ones(1, len(p_switch[rows_to_ceil, cols_to_ceil])))  # 1
        # p_switch_copy[rows_to_floor, cols_to_floor] = torch.minimum(p_switch[rows_to_floor, cols_to_floor], torch.zeros(1, len(p_switch[rows_to_floor, cols_to_floor])))  # 0

        # topology = p_switch_copy.flatten()
        # return topology

    def complete_JA(self, x, v, p_flow, topo, incidence):
        """
        return the completion variables to satisfy the power flow equations

        args:
            x: the input to the neural network
            v: the voltage magnitudes
            p_flow: the active power flows
            topo: the topology of the graph
            incidence: the incidence matrix of the graph

        return:
            pg: the active power generation
            qg: the reactive power generation
            p_flow_corrected: the active power flows corrected for the topology
            q_flow_corrected: the reactive power flows corrected for the topology
        """

        q_flow = (
            0.5 * (v.unsqueeze(1) @ incidence.float()).squeeze() - self.Rall * p_flow
        ) / self.Xall

        # TODO assert here that the equation is satisfied with vi-vj == qij + pij

        q_flow_corrected = topo * q_flow.float()
        p_flow_corrected = topo * p_flow

        pl = x[:, :, 0]
        ql = x[:, :, 1]

        pg = pl - (incidence.float() @ p_flow_corrected.unsqueeze(2)).squeeze()
        qg = ql - (incidence.float() @ q_flow_corrected.unsqueeze(2)).squeeze()

        return pg, qg, p_flow_corrected, q_flow_corrected

    def ineq_resid_JA(self, z, zc, idx, incidence):
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

        pij, v, topology = self.decompose_vars_z_JA(z)
        qij, pg, qg = self.decompose_vars_zc_JA(zc)

        pg_upp_resid = pg[:, 1::] - self.pgUpp[idx, :]
        pg_low_resid = self.pgLow[idx, :] - pg[:, 1::]

        qg_upp_resid = qg[:, 1::] - self.qgUpp[idx, :]
        qg_low_resid = self.qgLow[idx, :] - qg[:, 1::]

        v_upp_resid = v - self.vUpp
        v_low_resid = self.vLow - v

        # TODO discuss with Rabab
        connectivity = (
            1 - (torch.abs(incidence.float()) @ topology.unsqueeze(2)).squeeze()
        )

        resids = torch.cat(
            [
                pg_upp_resid,
                pg_low_resid,
                qg_upp_resid,
                qg_low_resid,
                pg[:, 0].reshape(-1, 1),
                -pg[:, 0].reshape(-1, 1),
                qg[:, 0].reshape(-1, 1),
                -qg[:, 0].reshape(-1, 1),
                v_upp_resid,
                v_low_resid,
                connectivity,
            ],
            dim=1,
        )

        violated_resid = torch.clamp(resids, min=0)
        # print(violated_resid[:,0])

        return violated_resid

    def cross_entropy_loss_topology(self, z, y, switch_mask):
        """
        cross_entropy_loss_topology returns the cross entropy loss between the topology chosen by the neural network and the reference topology

        args:
            z_hat: the output of the neural network
            y: the reference solution
            switch_mask: the mask for the switches

        return:
            loss: the cross entropy loss between the topology chosen by the neural network and the reference topology
        """

        _, opt_y_no_last, _, _, _, _, _, _, _ = self.decompose_vars_z(
            y[:, : self.zrdim]
        )
        _, y_last, _, _, _ = self.decompose_vars_zc(y[:, self.zrdim : :])

        y = torch.cat((opt_y_no_last, y_last.view(-1, 1)), dim=1)
        _, _, topology = self.decompose_vars_z_JA(z)
        switch_decision = topology[switch_mask].reshape_as(y)

        criterion = nn.BCELoss(reduction="sum")
        return criterion(switch_decision, y)

    def squared_error_topology(self, z, y, switch_mask):
        """
        squared_error_topology returns the squared error between the topology chosen by the neural network and the reference topology

        args:
            z_hat: the output of the neural network
            y: the reference solution
            switch_mask: the mask for the switches

        return:
            loss: the squared error between the topology chosen by the neural network and the reference topology
        """
        _, opt_y_no_last, _, _, _, _, _, _, _ = self.decompose_vars_z(
            y[:, : self.zrdim]
        )
        _, y_last, _, _, _ = self.decompose_vars_zc(y[:, self.zrdim : :])

        y = torch.cat((opt_y_no_last, y_last.view(-1, 1)), dim=1)
        _, _, topology = self.decompose_vars_z_JA(z)
        switch_decision = topology[switch_mask].reshape_as(y)

        criterion = nn.MSELoss(reduction="none")
        distance = criterion(switch_decision, y)
        return torch.sum(distance, dim=1)



    def opt_topology_dist_JA(self, z, y, switch_mask):
        """
        opt_dispatch_dist_JA returns the squared distance between the topology chosen by the neural network and the reference topology

        args:
            z_hat: the output of the neural network
            y: the reference solution
            switch_mask: the mask for the switches

        return:
            delta_topo: the squared distance between the topology chosen by the neural network and the reference topology
        """

        _, opt_y_no_last, _, _, _, _, _, _, _ = self.decompose_vars_z(
            y[:, : self.zrdim]
        )
        _, y_last, _, _, _ = self.decompose_vars_zc(y[:, self.zrdim : :])

        y = torch.cat((opt_y_no_last, y_last.view(-1, 1)), dim=1)

        _, _, topology = self.decompose_vars_z_JA(z)

        switch_decision = topology[switch_mask].reshape_as(y)
        delta_topo = torch.square(switch_decision - y)

        sum = torch.sum(delta_topo, dim=1)
        max = torch.max(torch.sum(delta_topo, dim=1))
        if torch.max(torch.sum(delta_topo, dim=1)) > 4:
            idx = torch.argmax(torch.sum(delta_topo, dim=1))
            print("problem")

        return delta_topo

    def opt_dispatch_dist_JA(self, z, zc, y):
        """
        opt_dispatch_dist_JA returns the squared distance between the output of
        the neural network and the completion variable from the reference solution

        args:
            z_hat: the output of the neural network
            zc_hat: the completion variables
            y: the reference solution
        return:
            dispatch_resid: the squared distance between the output of the neural network and the completion variable from the refernce solution
        """

        (
            _,
            opt_y_no_last,
            opt_pij,
            opt_pji,
            opt_qij,
            opt_qji_sw,
            opt_v,
            opt_plf,
            opt_qlf,
        ) = self.decompose_vars_z(y[:, : self.zrdim])
        _, y_last, opt_qji_nosw, opt_pg, opt_qg = self.decompose_vars_zc(
            y[:, self.zrdim : :]
        )
        opt_qji = torch.concat((opt_qji_nosw, opt_qji_sw), dim=1)
        opt_pg[:, 0] = opt_pg[:, 0] - opt_plf
        opt_qg[:, 0] = opt_qg[:, 0] - opt_qlf
        y = torch.cat((opt_y_no_last, y_last.view(-1, 1)), dim=1)

        pij, v, topology = self.decompose_vars_z_JA(z)
        qij, pg, qg = self.decompose_vars_zc_JA(zc)

        delta_pij = torch.square(pij - (opt_pij - opt_pji))
        delta_qij = torch.square(qij - (opt_qij - opt_qji))
        delta_v = torch.square(v[:, 1::] - opt_v).pow(2)
        delta_pg = torch.square(pg - opt_pg)
        delta_qg = torch.square(qg - opt_qg)
        # switch_decision = topology[switch_mask].reshape_as(y)
        # delta_topo = torch.square(switch_decision - y) / (
        #     2 * torch.sum(y, dim=1).view(-1, 1)
        # )

        # dist = torch.cat(
        #     (delta_pij, delta_qij, delta_v, delta_pg, delta_qg, delta_topo), dim=1
        # )
        dist = torch.cat([delta_v, delta_pg, delta_qg], dim=1)
        return dist

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


def total_loss(z, zc, criterion, utils, args, idx, incidence, train):
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
    ineq_dist = utils.ineq_resid_JA(
        z, zc, idx, incidence
    )  # gives update for vector weight
    # print(ineq_dist[0, :])
    ineq_cost = torch.linalg.vector_norm(
        ineq_dist, dim=1
    )  # gives norm for scalar weight

    soft_weight = args["softWeight"]

    total_loss = obj_cost + soft_weight * ineq_cost
    return total_loss, soft_weight


def opt_dispatch_dist(self, x, z, zc, idx, nn_mode):
    _, _, _, _, _, _, v, plf, qlf = self.decompose_vars_z(z)
    _, _, _, pg, qg = self.decompose_vars_zc(zc)

    # get the optimal data value
    if nn_mode == "train":
        opt_y = self.trainY[idx]
    elif nn_mode == "test":
        opt_y = self.testY[idx]
    elif nn_mode == "valid":
        opt_y = self.validY[idx]

    opt_z = opt_y[:, 0 : self.zdim]
    opt_zc = opt_y[:, self.zdim : self.zcdim + self.zdim]
    # self.zdim + (0:self.zcdim)

    _, _, _, _, _, _, opt_v, opt_plf, opt_qlf = self.decompose_vars_z(opt_z)
    _, _, _, opt_pg, opt_qg = self.decompose_vars_zc(opt_zc)

    dispatch_resid = torch.cat(
        [
            torch.square(v - opt_v),
            (torch.square(plf - opt_plf)).unsqueeze(1),
            (torch.square(qlf - opt_qlf)).unsqueeze(1),
            torch.square(pg - opt_pg),
            torch.square(qg - opt_qg),
        ],
        dim=1,
    )  # return squared error

    return dispatch_resid


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

    return defaults

class Modified_Sigmoid(nn.Module):
    def __init__(self, tau=5):
        super(Modified_Sigmoid, self).__init__()
        self.tau = tau

    def forward(self, p):
        return torch.clamp(2/(1+torch.exp(-self.tau*p)) - 1, 0)  # modified sigmoid