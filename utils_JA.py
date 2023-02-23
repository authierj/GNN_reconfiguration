import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function


class Utils:

    def __init__(self, data):

        self.A = data.A
        self.M = data.M
        self.N = data.N
        self.numSwitches = data.numSwitches
        self.pl = data.pl
        self.ql = data.ql
        self.Rall = data.Rall
        self.Xall = data.Xall
        self.pgUpp = data.pgUpp
        self.pgLow = data.pgLow
        self.qgUpp = data.qgUpp
        self.qgLow = data.qgLow
        self.bigM = data.bigM
        self.Xall = data.Xall
        self.Rall = data.Rall
        self.vUpp = data.vUpp
        self.vLow = data.vLow
        self.Aabs = data.Aabs
        self.dineq_dz_fullmat = data.dineq_dz_fullmat
        self.dzc_dz_mat = data.dzc_dz_mat
        self.dzc_dz = data.dzc_dz
        self.dzc_dx = data.dzc_dx

    def decompose_vars_z(self, z):
        """
        decompose_vars returns the decomposition of the neural network guess 

        :param z: the neural network guess
                    z = [zji, y\last, pij, pji, qji, qij_sw, v\f, plf, qlf]
        :return: the decomposition of z
        """

        zji = z[:, 0:self.M]
        y_nol = z[:, self.M + np.arange(0, self.numSwitches-1)]
        pij = z[:, self.M + self.numSwitches - 1 + np.arange(0, self.M)]
        pji = z[:, 2*self.M + self.numSwitches - 1 + np.arange(0, self.M)]
        qji = z[:, 3*self.M + self.numSwitches - 1 + np.arange(0, self.M)]
        qij_sw = z[:, 4*self.M + self.numSwitches - 1
                   + np.arange(0, self.numSwitches)]
        v = z[:, 4*self.M + 2*self.numSwitches - 1 +
              np.arange(0, self.N - 1)]  # feeder not included
        plf = z[:, 4*self.M + 2*self.numSwitches - 1 + self.N - 1]
        qlf = z[:, 4*self.M + 2*self.numSwitches - 1 + self.N]

        return zji, y_nol, pij, pji, qji, qij_sw, v, plf, qlf

    def decompose_vars_zc(self, zc):
        # zc = [zij, ylast, {qij}nosw, pg, qg]  # completion vars
        zij = zc[:, np.arange(0, self.M)]
        ylast = zc[:, self.M]
        qij_nosw = zc[:, self.M + 1 + np.arange(0, self.M - self.numSwitches)]
        pg = zc[:, 2 * self.M + 1 - self.numSwitches + np.arange(0, self.N)]
        qg = zc[:, 2 * self.M + 1 - self.numSwitches + self.N
                + np.arange(0, self.N)]

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
        _, _, pij, pji, qji, qij_sw,  _, _, _ = self.decompose_vars_z(z)
        _, _, qij_nosw, _, _ = self.decompose_vars_zc(zc)

        qij = torch.hstack((qij_nosw, qij_sw))

        fncval = torch.sum(torch.mul(torch.square(pij) + torch.square(pji)
                                     + torch.square(qij) + torch.square(qji), self.Rall), dim=1)

        return fncval

    def eq_resid(self, z, zc):
        # should be zero, but implementing it as a debugging step
        # pl, ql = self.decompose_vars_x(x)
        zji, y_nol, pij, pji, qji, qij_sw, v, plf, qlf = self.decompose_vars_z(
            z)
        zij, ylast, qij_nosw, pg, qg = self.decompose_vars_zc(zc)

        qij = torch.hstack((qij_nosw, qij_sw))
        y = torch.hstack((y_nol, ylast.unsqueeze(1)))

        ncases = z.shape[0]
        numsw = self.numSwitches
        vall = torch.hstack((torch.ones(ncases, 1), v))

        all_ohm_eqns = (torch.matmul(-vall, self.A.double())
                        + 2*(torch.mul((pij-pji), self.Rall)
                        + torch.mul((qij-qji), self.Xall)))

        # TODO split-up this equation
        resids = torch.cat([zij[:, 0:self.M - numsw] + zji[:, 0:self.M - numsw] - torch.ones(ncases, self.M-numsw),
                            zij[:, self.M - numsw:] +
                            zji[:, self.M - numsw:] - y,
                            (torch.sum(y, 1) - ((self.N - 1) -
                             (self.M - numsw))).unsqueeze(1),
                            # should this be pji - pij (equivalently reverse sign on the full term)
                            pg - (torch.hstack((plf.unsqueeze(1), self.pl)).T + \
                                  torch.mm(self.A.double(), torch.transpose(pij - pji, 0, 1))).T,
                            all_ohm_eqns[:, 0:self.M - numsw],
                            # should this be qji - qij (equivalently reverse sign on the full term)
                            qg - (torch.hstack((qlf.unsqueeze(1), self.ql)).T + \
                                  torch.mm(self.A.double(), torch.transpose(qij-qji, 0, 1))).T
                            ], dim=1)

        return resids

    def ineq_resid(self, z, zc, idx):  # Z = [z,zc]
        zji, y_nol, pij, pji, qji, qij_sw, v, plf, qlf = self.decompose_vars_z(
            z)
        zij, ylast, qij_nosw, pg, qg = self.decompose_vars_zc(zc)

        qij = torch.hstack((qij_nosw, qij_sw))
        y = torch.hstack((y_nol, ylast.unsqueeze(1)))

        ncases = z.shape[0]
        vall = torch.hstack((torch.ones(ncases, 1), v))

        pg_upp_resid = pg[:, 1:None] - self.pgUpp[idx, :]
        pg_low_resid = self.pgLow[idx, :] - pg[:, 1:None]

        qg_upp_resid = qg[:, 1:None] - self.qgUpp[idx, :]
        qg_low_resid = self.qgLow[idx, :] - qg[:, 1:None]

        v_upp_resid = v-self.vUpp
        v_low_resid = v-self.vLow

        # TODO rewrite these and find better names
        matrix1 = torch.mm(torch.neg(torch.transpose(torch.index_select(self.A, 1, torch.from_numpy(np.arange(self.M - self.numSwitches, self.M)).long()), 0, 1)),
                           vall.T.double()).T + 2 * (torch.mul((pij - pji), self.Rall) + torch.mul((qij - qji), self.Xall))[:, -self.numSwitches:None] - self.bigM * (1 - y)

        matrix2 = -torch.mm(torch.neg(torch.transpose(torch.index_select(self.A, 1, torch.from_numpy(np.arange(self.M - self.numSwitches, self.M)).long()), 0, 1)),
                            vall.T.double()).T - 2 * (torch.mul((pij - pji), self.Rall) + torch.mul((qij - qji), self.Xall))[:, -self.numSwitches:None] - self.bigM * (1 - y)

        # TODO rewrite that in multiple steps
        resids = torch.cat([pg_upp_resid, pg_low_resid, qg_upp_resid, qg_low_resid,
                            -torch.reshape(plf, (-1, 1)),
                            -torch.reshape(qlf, (-1, 1)),
                            -torch.reshape(pg[:, 0], (-1, 1)),
                            -torch.reshape(qg[:, 0], (-1, 1)), v_upp_resid,
                            v_low_resid, pij-self.bigM, -pij, pji-self.bigM,
                            -pji, qij-self.bigM, -qij, qji - self.bigM, -qji,
                            -zij, -zji, -y, matrix1, matrix2,
                            1 - torch.mm(self.Aabs.double(), (zij+zji).T).T
                            ], dim=1)

        return torch.clamp(resids, 0)

    def corr_steps(self, z, zc, idx):
        pos_resids = self.ineq_resid(z, zc, idx)
        delz = 2*torch.mm(pos_resids, self.dineq_dz_fullmat)
        delzc = torch.matmul(self.dzc_dz_mat, delz.T).T
        return delz, delzc

    def corr_steps_partial(self, z, zc, idx):
        pos_resids = self.ineq_resid(z, zc, idx)
        delz = 2*torch.mm(pos_resids, self.dineq_dz_partialmat)
        delzc = torch.matmul(self.dzc_dz_mat, delz.T).T
        return delz, delzc

    def mixed_sigmoid(self, nnout, epoch):
        # apply sigmoids here to get pg and v, and integer-ify the topology selection

        # parameter tau controls sharpness of the sigmoid function applied to binary variables. larger = sharper
        # tau = 5
        if epoch < 300:
            tau = 5
        # elif epoch < 800:
        #     tau = 10
        else:
            tau = 10

        z = sigFnc(self)(nnout, tau)

        # z = [zji, y\last, pij, pji, qji, {qij}sw, v\f, plf, qlf]  # initial guess from NN
        # zji and y\last are binary vars, which have already been processed to be between 0 and 1
        # process pij, pji, qji, {qij}_sw, v\f, plf, qlf to be within reasonable bounds
        zji, y_nol, pij_nn, pji_nn, qji_nn, qij_nosw_nn, v_nn, plf, qlf = self.decompose_vars_z(
            z)
        pij = pij_nn * self.bigM  # pij_min = 0; don't enforce any power flow constraints here
        pji = pji_nn * self.bigM  # pji_min = 0; don't enforce any power flow constraints here
        qji = qji_nn * self.bigM  # qji_min = 0; don't enforce any power flow constraints here
        qij_nosw = qij_nosw_nn * self.bigM
        v = v_nn*self.vUpp + (1-v_nn)*self.vLow
        z_physical = torch.hstack(
            (zji, y_nol, pij, pji, qji, qij_nosw, v, plf.unsqueeze(1), qlf.unsqueeze(1)))

        return z_physical

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
        zji, y_nol, pij_nn, pji_nn, qji_nn, qij_nosw_nn, v_nn, plf, qlf = self.decompose_vars_z(
            z)
        pij = pij_nn * self.bigM  # pij_min = 0; don't enforce any power flow constraints here
        pji = pji_nn * self.bigM  # pji_min = 0; don't enforce any power flow constraints here
        qji = qji_nn * self.bigM  # qji_min = 0; don't enforce any power flow constraints here
        qij_nosw = qij_nosw_nn * self.bigM
        v = v_nn*self.vUpp + (1-v_nn)*self.vLow
        z_physical = torch.hstack(
            (zji, y_nol, pij, pji, qji, qij_nosw, v, plf.unsqueeze(1), qlf.unsqueeze(1)))

        return z_physical

    def mixedIntOutput(self, z, tau):
        bin_vars_zji = z[:, 0:self.M]
        bin_vars_y = z[:, self.M:(self.M + self.numSwitches-1)]
        cont_vars = z[:, self.M+self.numSwitches-1:None]
        # classical sigmoid for continuous variables
        output_cont = nn.Sigmoid()(cont_vars)

        # modified sigmoid
        output_bin_zji = torch.clamp(2/(1+torch.exp(-tau*bin_vars_zji)) - 1, 0)

        batchsize = z.size(dim=0)
        # r = np.random.randint(-1, 1, batchsize)

        # PHYSIC INFORMED ROUNDING

        L_min = (self.N - 1) - (self.M - self.numSwitches)
        y_sorted_inds = torch.argsort(bin_vars_y)  # sorted in ascending order

        # output_bin_y = bin_vars_y.abs()
        # output_bin_y = nn.Sigmoid()(bin_vars_y) # cast between 0 and 1
        output_bin_y = torch.clamp(bin_vars_y, 0, 1)
        # ceil the largest L values to 1, floor the smallest size(bin_y)-L values to 0
        rows_to_ceil = np.hstack((np.arange(0, batchsize).repeat(L_min),
                                  ))  # np.where(r == 0)[0]
        cols_to_ceil = np.hstack((y_sorted_inds[:, -L_min:].flatten(),
                                  ))  # y_sorted_inds[np.where(r == 0)[0], -L_min-1]

        num_to_zero = bin_vars_y.size(dim=1) - L_min - 1
        if num_to_zero < 0:
            num_to_zero = 0
        rows_to_floor = np.hstack((np.arange(0, batchsize).repeat(num_to_zero),
                                   ))  # np.where(r == -1)[0]
        cols_to_floor = np.hstack((y_sorted_inds[:, 0:num_to_zero].flatten(),
                                   ))  # y_sorted_inds[np.where(r == -1)[0], -L_min-1]

        # output_bin_y[rows_to_ceil, cols_to_ceil] = output_bin_y[rows_to_ceil, cols_to_ceil].ceil()
        # output_bin_y[rows_to_floor, cols_to_floor] = output_bin_y[rows_to_floor, cols_to_floor].floor()
        output_bin_y[rows_to_ceil, cols_to_ceil] = 1
        output_bin_y[rows_to_floor, cols_to_floor] = 0

        z_new = torch.hstack((output_bin_zji, output_bin_y, output_cont))
        return z_new

    def complete(self, x, z):  # , zslack
        z, zc = dc3Function(self)(x, z)
        return z, zc

    def decompose_vars_x(self, x):
        # x = [pl\f, ql\f]  # input to NN
        pl = x[:, 0:self.N - 1]
        ql = x[:, self.N - 1:None]
        return pl, ql


def dc3Function(data):
    class dc3FunctionFn(Function):
        @staticmethod
        def forward(ctx, x, z):
            # this will perform the completion step
            pl, ql = data.decompose_vars_x(x)
            zji, y_nol, pij, pji, qji, qij_sw, v, plf, qlf = data.decompose_vars_z(
                z)
            numsw = data.numSwitches
            ncases = x.shape[0]

            y_rem = (data.N - 1) - (data.M - numsw) - torch.sum(y_nol, 1)
            zij = torch.zeros(ncases, data.M)
            zij[:, 0:data.M - numsw] = 1 - zji[:, 0:data.M - numsw]
            zij[:, data.M - numsw:-1] = y_nol - zji[:, data.M - numsw:-1]
            zij[:, -1] = y_rem - zji[:, -1]

            pg = torch.hstack((plf.unsqueeze(1), pl)).T.double(
            ) + torch.mm(data.A.double(), torch.transpose(pij-pji, 0, 1).double())

            # TODO check if this is implemented correctly with the A*v step - should be good
            vall = torch.hstack((torch.ones(ncases, 1), v))
            delQ = torch.div((-0.5 * torch.matmul(-vall.double(), data.A.double()
                                                  ) - torch.mul((pij-pji).double(), data.Rall)), data.Xall)
            delQ_nosw = delQ[:, 0:-numsw]

            qij_rem = delQ_nosw + qji[:, 0:-numsw]
            qij = torch.hstack((qij_rem, qij_sw))

            # TODO: Feb 1, 2022: check if this is correct
            # Feb 1, 2022 edit on last term; was qji[:, numsw:])
            delQ = torch.hstack((delQ_nosw, qij_sw - qji[:, data.M-numsw:]))
            qg = (torch.hstack((qlf.unsqueeze(1), ql)).T +
                  torch.mm(data.A.double(), torch.transpose(delQ, 0, 1)))

            zc = torch.hstack((zij, y_rem.unsqueeze(1), qij_rem, pg.T, qg.T))
            zc.requires_grad = True

            return z, zc

        @staticmethod
        def backward(ctx, dl_dz, dl_dzc):
            # data = ctx.data
            dl_dz = dl_dz + torch.matmul(dl_dzc, data.dzc_dz)
            dl_dx = torch.matmul(dl_dzc, data.dzc_dx)

            return dl_dx, dl_dz

    return dc3FunctionFn.apply


def sigFnc(data):
    class MixedSigmoid(Function):
        # Sigmoids applied at the last layer of the NN
        # includes sigmoid approximation for indicator function, used for binary variable selection
        # Reference: page 7 of https://arxiv.org/abs/2004.02402
        @staticmethod
        def forward(ctx, z, tau):
            bin_vars = z[:, 0:data.M+data.numSwitches-1]
            cont_vars = z[:, data.M+data.numSwitches-1:None]
            output_cont = nn.Sigmoid()(cont_vars)
            # output_bin = nn.Sigmoid()(bin_vars)
            output_bin = torch.clamp(2/(1+torch.exp(-tau*bin_vars)) - 1, 0)
            z_new = torch.hstack((output_bin, output_cont))

            ctx.save_for_backward(z_new)
            ctx.data = data
            ctx.tau = tau
            return z_new

        @staticmethod
        def backward(ctx, grad_output):
            z, = ctx.saved_tensors
            data = ctx.data
            tau = ctx.tau

            # what is clone() used for?
            bin_grad = grad_output[:, 0:data.M + data.numSwitches - 1].clone()
            cont_grad = grad_output[:, data.M +
                                    data.numSwitches - 1:None].clone()
            bin_vars = z[:, 0:data.M + data.numSwitches - 1]
            cont_vars = z[:, data.M + data.numSwitches - 1:None]

            # derivative of custom binary approx sigmoid
            bin_grad[bin_vars < 0] = 0  # for the first part of the function
            exp_res = torch.exp(-tau * bin_vars)
            inter_res = 2 * tau * exp_res / torch.square(1 + exp_res)
            bin_grad[bin_vars >= 0] *= inter_res[bin_vars >= 0]

            # bin_grad *= torch.exp(-bin_vars) / torch.square(1 + torch.exp(-bin_vars))

            # derivative of traditional sigmoid - do we need to do this explicitly?
            cont_grad *= torch.exp(-cont_vars) / \
                torch.square(1 + torch.exp(-cont_vars))

            # grad_input = grad_output.clone()
            grad_input = torch.hstack((bin_grad, cont_grad))

            return grad_input, None

    return MixedSigmoid.apply


def mixedIntFnc(data):
    class MixedIntOutput(Function):
        # Output layer of NN
        # zji: sigmoid approximation for indicator function, used for binary variable selection
        # y\last: rounding
        # other vars: standard sigmoid
        # Reference sig approx: page 7 of https://arxiv.org/abs/2004.02402
        @staticmethod
        def forward(ctx, z, tau):
            bin_vars_zji = z[:, 0:data.M]
            bin_vars_y = z[:, data.M:(data.M + data.numSwitches-1)]
            cont_vars = z[:, data.M+data.numSwitches-1:None]
            output_cont = nn.Sigmoid()(cont_vars)
            # output_bin = nn.Sigmoid()(bin_vars)

            output_bin_zji = torch.clamp(
                2/(1+torch.exp(-tau*bin_vars_zji)) - 1, 0)

            batchsize = z.size(dim=0)
            r = np.random.randint(-1, 1, batchsize)
            L_min = (data.N - 1) - (data.M - data.numSwitches)
            # sorted in ascending order
            y_sorted_inds = torch.argsort(bin_vars_y)

            output_bin_y_test = bin_vars_y.abs()
            # ceil the largest L values to 1, floor the smallest size(bin_y)-L values to 0
            rows_to_ceil = np.hstack((np.arange(0, batchsize).repeat(L_min),
                                      np.where(r == 0)[0]))
            cols_to_ceil = np.hstack((y_sorted_inds[:, -L_min:].flatten(),
                                      y_sorted_inds[np.where(r == 0)[0], -L_min-1]))

            num_to_zero = bin_vars_y.size(dim=1) - L_min - 1
            rows_to_floor = np.hstack((np.arange(0, batchsize).repeat(num_to_zero),
                                       np.where(r == -1)[0]))
            cols_to_floor = np.hstack((y_sorted_inds[:, 0:num_to_zero].flatten(),
                                       y_sorted_inds[np.where(r == -1)[0], -L_min-1]))

            output_bin_y_test[rows_to_ceil,
                              cols_to_ceil] = output_bin_y_test[rows_to_ceil, cols_to_ceil].ceil()
            output_bin_y_test[rows_to_floor,
                              cols_to_floor] = output_bin_y_test[rows_to_floor, cols_to_floor].floor()

            z_new = torch.hstack(
                (output_bin_zji, output_bin_y_test, output_cont))

            ctx.save_for_backward(z_new)
            ctx.data = data
            ctx.tau = tau
            return z_new

        @staticmethod
        def backward(ctx, grad_output):
            z, = ctx.saved_tensors
            data = ctx.data
            tau = ctx.tau

            # what is clone() used for?
            bin_grad = grad_output[:, 0:data.M + data.numSwitches - 1].clone()
            cont_grad = grad_output[:, data.M +
                                    data.numSwitches - 1:None].clone()
            bin_vars = z[:, 0:data.M + data.numSwitches - 1]
            cont_vars = z[:, data.M + data.numSwitches - 1:None]

            # derivative of custom binary approx sigmoid
            bin_grad[bin_vars < 0] = 0  # for the first part of the function
            exp_res = torch.exp(-tau * bin_vars)
            inter_res = 2 * tau * exp_res / torch.square(1 + exp_res)
            bin_grad[bin_vars >= 0] *= inter_res[bin_vars >= 0]

            # bin_grad *= torch.exp(-bin_vars) / torch.square(1 + torch.exp(-bin_vars))

            # derivative of traditional sigmoid - do we need to do this explicitly?
            cont_grad *= torch.exp(-cont_vars) / \
                torch.square(1 + torch.exp(-cont_vars))

            # grad_input = grad_output.clone()
            grad_input = torch.hstack((bin_grad, cont_grad))

            return grad_input, None

    return MixedIntOutput.apply


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
    graph_3d = torch.reshape(x_graph, (batch_size,
                                       int(x_graph.shape[0]/batch_size),
                                       x_graph.shape[1]))

    if not first_node:
        graph_3d = graph_3d[:, 1::, :]
    xNN_3d = torch.transpose(graph_3d, 1, 2)
    xNN = torch.flatten(xNN_3d, 1, 2)
    return xNN
