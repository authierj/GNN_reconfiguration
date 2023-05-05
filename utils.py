from setup import *
import sympy as sym
import scipy.sparse as sp
import scipy.io as spio
import scipy.linalg as lnalg
import numpy as np
import random

import hashlib

import torch
import torch.nn as nn
from torch.autograd import Function

DEVICE = torch.device("cpu")  # torch.device("cuda") if torch.cuda.is_available() else


#### From DC3 codebase

def my_hash(string):
    return hashlib.sha1(bytes(string, 'utf-8')).hexdigest()
####

class OPTreconfigure():
    def __init__(self, filename, dataflag='partial', valid_frac=0.1, test_frac=0.1):
        # load network data by name... - replace this function
        data = spio.loadmat(filename)

        cases = data['case_data_all'][0][0]
        pl = cases[0]
        ql = cases[1]
        pgLow = cases[2]
        pgUpp = cases[3]
        qgLow = cases[4]
        qgUpp = cases[5]

        network = data['network_data'][0, 0] # network = data['network_33_data'][0, 0]
        SBase = np.squeeze(network[0])
        VBase = np.squeeze(network[1])
        ZBase = np.squeeze(network[2])
        YBase = np.squeeze(network[3])
        IBase = np.squeeze(network[4])
        N = np.squeeze(network[5])
        M = np.squeeze(network[6])
        numFeeders = np.squeeze(network[7])  # new for the 83 node network
        swInds = np.squeeze(network[8])
        numSwitches = np.squeeze(network[9])
        mStart = network[10]
        mEnd = network[11]
        Rall = np.squeeze(network[14])
        Xall = np.squeeze(network[15])
        vLow = np.square(np.squeeze(network[16]))  # not a vector, just a value
        vUpp = np.square(np.squeeze(network[17]))  # not a vector, just a value
        bigM = 0.5  # all quantities in pu, so 1pu = SBase

        # network data
        self.N = N.item(0)
        self.M = M.item(0)
        self.SBase = SBase.item(0)
        self.numFeeders = numFeeders.item(0)

        # Adjust to all switches are at the end (setup.py needs all switches are at the end. Matlab code does not)
        self.swInds = swInds  # doesn't need to be a tensor bc an index
        self.numSwitches = numSwitches.item(0)
        interim = list(set(np.arange(0, M)) ^ set(self.swInds - 1))  # non-switch lines
        self.Rall = torch.cat((torch.from_numpy(Rall[interim]), torch.from_numpy(Rall[swInds - 1])))
        self.Xall = torch.cat((torch.from_numpy(Xall[interim]), torch.from_numpy(Xall[swInds - 1])))
        mStart_reshaped = sp.hstack((mStart[:, interim], mStart[:, swInds - 1]))
        mEnd_reshaped = sp.hstack((mEnd[:, interim], mEnd[:, swInds - 1]))
        Acoo = mStart_reshaped - mEnd_reshaped

        mStart_tensor = torch.sparse_coo_tensor(
            torch.vstack((torch.from_numpy(mStart_reshaped.tocoo().row), torch.from_numpy(mStart_reshaped.tocoo().col))),
            mStart_reshaped.data, torch.Size(mStart_reshaped.shape))  # indices, values, size
        mEnd_tensor = torch.sparse_coo_tensor(
            torch.vstack((torch.from_numpy(mEnd_reshaped.tocoo().row), torch.from_numpy(mEnd_reshaped.tocoo().col))),
            mEnd_reshaped.data, torch.Size(mEnd_reshaped.shape))  # indices, values, size

        # mStart_tensor = torch.sparse_coo_tensor(
        #     torch.vstack((torch.from_numpy(
        #         np.concatenate((mStart[:, interim].tocoo().row, mStart[:, self.swInds - 1].tocoo().row))),
        #                   torch.arange(0, self.M))),  # all cols should be 0:M-1
        #     np.concatenate((mStart.data[interim], mStart.data[self.swInds - 1])),
        #     torch.Size(mStart.shape))  # indices, values, size
        #
        # mEnd_tensor = torch.sparse_coo_tensor(
        #     torch.vstack((torch.from_numpy(
        #         np.concatenate((mEnd[:, interim].tocoo().row, mEnd[:, self.swInds - 1].tocoo().row))),
        #                   torch.arange(0, self.M))),
        #     np.concatenate((mEnd.data[interim], mEnd.data[self.swInds - 1])),
        #     torch.Size(mEnd.shape))  # indices, values, size

        self.mStart = mStart_tensor
        self.mEnd = mEnd_tensor
        self.A = (mStart_tensor - mEnd_tensor).to_dense()
        self.Aabs = mStart_tensor + mEnd_tensor



        # Pytorch want each row a case, each column a node
        # randomize the data so different cases appear in training, validation, testing
        perm = np.arange(pl.shape[1])
        np.random.shuffle(perm)  # in place
        # specific load + generation data
        if pl.shape[0] == self.N:
            self.pl = torch.from_numpy(pl[1:, perm].T)  # pl and ql given for all nodes including first feeder
            self.ql = torch.from_numpy(ql[1:, perm].T)
            self.pgUpp = torch.from_numpy(pgUpp[1:, perm].T)  # all limits including first feeder
            self.pgLow = torch.from_numpy(pgLow[1:, perm].T)
            self.qgUpp = torch.from_numpy(qgUpp[1:, perm].T)
            self.qgLow = torch.from_numpy(qgLow[1:, perm].T)
        else:
            self.pl = torch.from_numpy(pl[:, perm].T)  # pl and ql given for all nodes except first feeder
            self.ql = torch.from_numpy(ql[:, perm].T)
            self.pgUpp = torch.from_numpy(pgUpp[:, perm].T)  # all limits excluding first feeder
            self.pgLow = torch.from_numpy(pgLow[:, perm].T)
            self.qgUpp = torch.from_numpy(qgUpp[:, perm].T)
            self.qgLow = torch.from_numpy(qgLow[:, perm].T)

        self.vUpp = torch.tensor(vUpp, dtype=torch.get_default_dtype())  # scalar value
        self.vLow = torch.tensor(vLow, dtype=torch.get_default_dtype())  # scalar value
        self.bigM = bigM

        # matrices for correction steps and backpropagation
        dineq_dz_mat = dineq_dz(N, M, swInds, Rall, Xall, bigM, Acoo).todense()
        dineq_dzc_mat = dineq_dzc(N, M, swInds, bigM, Acoo).todense()
        dzc_dz_mat, dzc_dx_mat = dzc_vars(N, M, swInds, Acoo, Rall, Xall)
        dineq_dz_fullmat = dineq_dz_mat + np.matmul(dineq_dzc_mat, dzc_dz_mat)
        self.dzc_dz_mat = torch.from_numpy(dzc_dz_mat)
        self.dzc_dx_mat = torch.from_numpy(dzc_dx_mat)
        self.dineq_dz_fullmat = torch.from_numpy(dineq_dz_fullmat)
        self.dineq_dz_partialmat = torch.from_numpy(dineq_dz_mat)


        # dzc_dz_mat_tensor = torch.sparse_coo_tensor(
        #     torch.vstack((torch.from_numpy(dzc_dz_mat.row), torch.from_numpy(dzc_dz_mat.col))), dzc_dz_mat.data,
        #     torch.Size(dzc_dz_mat.shape))  # indices, values, size
        # dzc_dx_mat_tensor = torch.sparse_coo_tensor(
        #     torch.vstack((torch.from_numpy(dzc_dx_mat.row), torch.from_numpy(dzc_dx_mat.col))), dzc_dx_mat.data,
        #     torch.Size(dzc_dx_mat.shape))  # indices, values, size
        # dineq_dz_fullmat_tensor = torch.sparse_coo_tensor(
        #     torch.vstack((torch.from_numpy(dineq_dz_fullmat.row), torch.from_numpy(dineq_dz_fullmat.col))), dineq_dz_fullmat.data,
        #     torch.Size(dineq_dz_fullmat.shape))  # indices, values, size

        # self.dineq_dz_mat = dineq_dz_mat
        # self.dineq_dzc_mat = dineq_dzc_mat
        # self.dzc_dz_mat = dzc_dz_mat
        # self.dzc_dx_mat = dzc_dx_mat
        # self.dineq_dz_fullmat = dineq_dz_mat + dineq_dzc_mat * dzc_dz_mat

        # self.dineq_dz_mat = dineq_dz_mat
        # self.dineq_dzc_mat = dineq_dzc_mat
        # self.dzc_dz_mat = dzc_dz_mat_tensor
        # self.dzc_dx_mat = dzc_dx_mat_tensor
        # self.dineq_dz_fullmat = dineq_dz_fullmat_tensor

        # optimization input and output variables
        # x = [pl\f, ql\f]  # input to NN
        # z = [zji, y\last, pij, pji, qji, {qij}sw, v\f, plf, qlf]  # initial guess from NN
        # zc = [zij, ylast, {qij}nosw, pg, qg]  # completion vars
        # y = [z, zc]

        # this is all the data - then it'll get split into train-test-valid
        x = np.concatenate([self.pl.T, self.ql.T])  # of x: each row is a node, each column a test case
        if dataflag.lower() == 'full':
            solutions = data['res_data_all'][0][0]
            # z, zc, yalmiptime, solvertime, objVal
            # pg, qg, v, yij, zij, zji, pij, pji, qij, qji = get_solution_data()
            # z = np.concatenate([zji, yij[0:-1], pij, pji, qji, qij[-numSwitches:None], v[1:None], [pl[0, :]], [ql[0, :]]])
            # zc = np.concatenate([zij, [yij[-1, :]], qij[0:M-numSwitches], pg, qg])
            z = solutions[0]
            zc = solutions[1]
            objvals = solutions[3]  # previously solutions[4] for node_4

            self._z = torch.from_numpy(z[:, perm].T)
            self._zc = torch.from_numpy(zc[:, perm].T)
            self._y = torch.hstack((self._z, self._zc))
            self._objval = torch.from_numpy(objvals[:, perm].T)
        else:
            self._z = None
            self._zc = None
            self._y = None
            self._objval = None

        # remove last row of PL and QL for networks with multiple feeders
        # Fixing the multiple feeder network design
        # if self.numFeeders > 1 and len(x) != 2*(N-1):
        #     x = np.concatenate([pl[:-1, :], ql[:-1, :]])
        #     self._x = torch.t(torch.tensor(x, dtype=torch.get_default_dtype()))
        # else:
        #     self._x = torch.t(torch.tensor(x, dtype=torch.get_default_dtype()))

        self._x = torch.t(torch.tensor(x, dtype=torch.get_default_dtype()))
        self._known = torch.t(torch.tensor(1, dtype=torch.get_default_dtype()))

        self._xdim = 2*(N-1)  # x       TODO: change this to 2*(N-numFeeders)?
        self._zdim = 4*M+2*numSwitches+N  # only z
        self._zcdim = 2*M+1-numSwitches+2*N  # only zc
        self._ydim = self._zdim + self._zcdim  # z, zc
        self._num = self._x.shape[0]  # number of test cases
        self._neq = 2*M+1-numSwitches+2*N  # equalities
            # M+1+N+N+M-#sw+M = 2*M + 1 + 2*N - numSwitches
        self._nineq = 6*(N-1) + 4 + 10*M + 3*numSwitches + N  # inequalities
        self._nknowns = 1  # number of known vars (and not x) vfeed

        ## Define train/valid/test split
        self._valid_frac = valid_frac
        self._test_frac = test_frac

        ### For Pytorch
        self._device = None

    def __str__(self):
        return 'NETWORK-{}-{}-{}'.format(
            self.N, self.valid_frac, self.test_frac)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def zc(self):
        return self._zc

    @property
    def obj_val(self):
        return self._objval

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def zdim(self):
        return self._zdim

    @property
    def zcdim(self):
        return self._zcdim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def dzc_dz(self):
        return self.dzc_dz_mat

    @property
    def dzc_dx(self):
        return self.dzc_dx_mat

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def valid_idx(self):
        startind = int(self.num * self.train_frac)
        endind = int(self.num * (self.train_frac + self.valid_frac))
        return startind, endind

    @property
    def test_idx(self):
        startind = int(self.num * (self.train_frac + self.valid_frac))
        endind = None
        return startind, endind

    @property
    def train_idx(self):
        startind = 0
        endind = int(self.num * self.train_frac)
        return startind, endind

    @property
    def trainX(self):
        s, e = self.train_idx
        return self.x[s:e]

    @property
    def validX(self):
        s, e = self.valid_idx
        return self.x[s:e]

    @property
    def testX(self):
        s, e = self.test_idx
        return self.x[s:e]

    @property
    def trainY(self):
        res = self.y
        if res is not None:
            return res[:int(self.num * self.train_frac)]
        else:
            return None

    @property
    def validY(self):
        res = self.y
        if res is not None:
            return res[int(self.num * self.train_frac):int(self.num * (self.train_frac + self.valid_frac))]
        else:
            return None

    @property
    def testY(self):
        # res = None
        # try:
        #     res = self.y[int(self.num * (self.train_frac + self.valid_frac)):]
        # except ValueError:
        #     print('Solution data not loaded for testY')
        # return res
        res = self.y
        if res is not None:
            return res[int(self.num * (self.train_frac + self.valid_frac)):]
        else:
            return None
            # raise ValueError

    @property
    def device(self):
        return self._device

    def decompose_vars_x(self, x):
        # x = [pl\f, ql\f]  # input to NN
        pl = x[:, 0:self.N - 1]
        ql = x[:, self.N - 1:None]
        return pl, ql

    def decompose_vars_z(self, z):
        # z = [zji, y\last, pij, pji, qji, qij_sw, v\f, plf, qlf]  # initial guess from NN

        zji = z[:, 0:self.M]
        y_nol = z[:, self.M + np.arange(0, self.numSwitches-1)]
        pij = z[:, self.M+self.numSwitches-1 + np.arange(0, self.M)]
        pji = z[:, 2*self.M+self.numSwitches-1 + np.arange(0, self.M)]
        qji = z[:, 3*self.M+self.numSwitches-1 + np.arange(0, self.M)]
        qij_sw = z[:, 4 * self.M + self.numSwitches - 1 + np.arange(0, self.numSwitches)]
        v = z[:, 4 * self.M + 2 * self.numSwitches - 1 + np.arange(0, self.N - 1)]  # last feeder not included
        plf = z[:, 4 * self.M + 2 * self.numSwitches - 1 + self.N - 1]
        qlf = z[:, 4 * self.M + 2 * self.numSwitches - 1 + self.N]
        # v = z[:, 4*self.M+self.numSwitches-1 + np.arange(0, self.N-1)]  # feeder not included
        # plf = z[:, 4*self.M+self.numSwitches-1+self.N-1]
        # qlf = z[:, 4*self.M+self.numSwitches-1+self.N]
        return zji, y_nol, pij, pji, qji, qij_sw, v, plf, qlf

    def decompose_vars_zc(self, zc):
        # zc = [zij, ylast, {qij}nosw, pg, qg]  # completion vars
        zij = zc[:, np.arange(0, self.M)]
        ylast = zc[:, self.M]
        qij_nosw = zc[:, self.M + 1 + np.arange(0, self.M - self.numSwitches)]
        pg = zc[:, 2 * self.M + 1 - self.numSwitches + np.arange(0, self.N)]
        qg = zc[:, 2 * self.M + 1 - self.numSwitches + self.N + np.arange(0, self.N)]
        # qij = zc[:, self.M+1 + np.arange(0, self.M)]
        # pg = zc[:, 2*self.M+1 + np.arange(0, self.N)]
        # qg = zc[:, 2*self.M+1+self.N + np.arange(0, self.N)]
        return zij, ylast, qij_nosw, pg, qg

    def get_integer_z(self, z):
        z_ij, y_nol, _, _, _, _, _, _, _ = self.decompose_vars_z(z)

        return torch.cat([z_ij, y_nol], dim = 1)

    def obj_fnc(self, x, z, zc):
        # # objective function: approximate power loss via line losses
        # # Rij * (Pij^2 + Pji^2 + Qij^2 + Qji^2)
        _, _, pij, pji, qji, qij_sw,  _, _, _ = self.decompose_vars_z(z)
        _, _, qij_nosw, _, _ = self.decompose_vars_zc(zc)

        qij = torch.hstack((qij_nosw, qij_sw))

        fncval = torch.sum(torch.mul(torch.square(pij) + torch.square(pji) + torch.square(qij) + torch.square(qji), self.Rall), dim=1)

        return fncval

    def opt_dispatch_dist(self, x, z, zc, idx, nn_mode):
        _, _, _, _, _, _, v, plf, qlf = self.decompose_vars_z(z)
        _, _, _, pg, qg = self.decompose_vars_zc(zc)

        # get the optimal data value
        if nn_mode == 'train':
            opt_y = self.trainY[idx]
        elif nn_mode == 'test':
            opt_y = self.testY[idx]
        elif nn_mode == 'valid':
            opt_y = self.validY[idx]

        opt_z = opt_y[:, 0:self.zdim]
        opt_zc = opt_y[:, self.zdim:self.zcdim+self.zdim]
        # self.zdim + (0:self.zcdim)

        _, _, _, _, _, _, opt_v, opt_plf, opt_qlf = self.decompose_vars_z(opt_z)
        _, _, _, opt_pg, opt_qg = self.decompose_vars_zc(opt_zc)

        dispatch_resid = torch.cat([ torch.square(v-opt_v),
                                     (torch.square(plf - opt_plf)).unsqueeze(1),
                                     (torch.square(qlf - opt_qlf)).unsqueeze(1),
                                     torch.square(pg - opt_pg),
                                     torch.square(qg - opt_qg)], dim=1) # return squared error

        return dispatch_resid

    def opt_topology_dist(self, x, z, zc, idx, nn_mode):
        _, y_nol, _, _, _, _, _, _, _ = self.decompose_vars_z(z)
        _, ylast, _, _, _ = self.decompose_vars_zc(zc)

        # get the optimal data value
        if nn_mode == 'train':
            opt_y = self.trainY[idx]
        elif nn_mode == 'test':
            opt_y = self.testY[idx]
        elif nn_mode == 'valid':
            opt_y = self.validY[idx]

        opt_z = opt_y[:, 0:self.zdim]
        opt_zc = opt_y[:, self.zdim:self.zcdim+self.zdim]

        _, opt_y_nol, _, _, _, _, _, _, _ = self.decompose_vars_z(opt_z)
        _, opt_ylast, _, _, _ = self.decompose_vars_zc(opt_zc)

        topology_resid = torch.cat([ torch.square(opt_y_nol - y_nol),
                                     (torch.square(opt_ylast - ylast)).unsqueeze(1)], dim=1)

        return topology_resid

    def train_adjustIDX(self, idx):
        start, _ = self.train_idx
        return start + idx

    def test_adjustIDX(self, idx):
        start, _ = self.test_idx
        return start + idx

    def valid_adjustIDX(self, idx):
        start, _ = self.valid_idx
        return start + idx

    def ineq_dist(self, z, zc, idx):
        return self.ineq_resid(z, zc, idx)

    def eq_resid(self, x, z, zc):
        # should be zero, but implementing it as a debugging step
        pl, ql = self.decompose_vars_x(x)
        zji, y_nol, pij, pji, qji, qij_sw, v, plf, qlf = self.decompose_vars_z(z)
        zij, ylast, qij_nosw, pg, qg = self.decompose_vars_zc(zc)

        qij = torch.hstack((qij_nosw, qij_sw))
        y = torch.hstack((y_nol, ylast.unsqueeze(1)))

        ncases = z.shape[0]
        numsw = self.numSwitches

        # Fixing the multiple feeder network design
        # if self.numFeeders > 1:
        #     vall = torch.hstack((v, torch.ones(ncases, 1)))  # multiple feeders (last few nodes). slack bus is last node
        #     pl_stacked = torch.hstack((pl, plf.unsqueeze(1)))
        #     ql_stacked = torch.hstack((ql, qlf.unsqueeze(1)))
        # else:
        #     vall = torch.hstack((torch.ones(ncases, 1), v))  # 1 feeder. slack bus is first node
        #     pl_stacked = torch.hstack((plf.unsqueeze(1), pl))
        #     ql_stacked = torch.hstack((qlf.unsqueeze(1), ql))

        vall = torch.hstack((torch.ones(ncases, 1), v))  # 1 feeder. slack bus is first node
        pl_stacked = torch.hstack((plf.unsqueeze(1), pl))
        ql_stacked = torch.hstack((qlf.unsqueeze(1), ql))
        # zij + zji = 1
        # zij + zji = y
        # sum(y) = (N-1) - (M-numsw)
        # y_rem = (data.N - 1) - (data.M - numsw) - torch.sum(y_nol, 1)
        # pg - pl = sum(pji - pij)
        # vj - vi = -2(Rij(Pij-Pji) + Xij(Qij-Qji)) -> should this be nonswitch lines only
        # qg - ql = sum(qji - qij)

        all_ohm_eqns = torch.matmul(-vall, self.A.double()) + 2*(torch.mul((pij-pji), self.Rall) + torch.mul((qij-qji), self.Xall))

        resids = torch.cat([zij[:, 0:self.M - numsw] + zji[:, 0:self.M - numsw] - torch.ones(ncases, self.M-numsw),
                            zij[:, self.M - numsw:] + zji[:, self.M - numsw:] - y,
                            (torch.sum(y, 1) - ((self.N - 1) - (self.M - numsw))).unsqueeze(1),
                            pg - (pl_stacked.T + torch.mm(self.A.double(), torch.transpose(pij - pji, 0, 1))).T,
                            all_ohm_eqns[:, 0:self.M - numsw],
                            qg - (ql_stacked.T + torch.mm(self.A.double(), torch.transpose(qij-qji, 0, 1))).T
                            ], dim=1)

        return resids

    def ineq_resid(self, z, zc, idx):  # Z = [z,zc]
        zji, y_nol, pij, pji, qji, qij_sw, v, plf, qlf = self.decompose_vars_z(z)
        zij, ylast, qij_nosw, pg, qg = self.decompose_vars_zc(zc)

        qij = torch.hstack((qij_nosw, qij_sw))
        y = torch.hstack((y_nol, ylast.unsqueeze(1)))

        ncases = z.shape[0]

        # Fixing the multiple feeder network design
        # if self.numFeeders > 1:
        #     vall = torch.hstack((v, torch.ones(ncases, 1)))  # multiple feeders (last few nodes). slack bus is last node
        # else:
        #     vall = torch.hstack((torch.ones(ncases, 1), v))  # 1 feeder. slack bus is first node

        vall = torch.hstack((torch.ones(ncases, 1), v))  # 1 feeder. slack bus is first node

        if self.pgUpp.shape[1] == self.N:
            resids = torch.cat([
                pg - self.pgUpp[idx, :],
                self.pgLow[idx, :] - pg,
                qg - self.qgUpp[idx, :],
                self.qgLow[idx, :] - qg,
                - torch.reshape(plf, (-1, 1)),
                - torch.reshape(qlf, (-1, 1)),
                v - self.vUpp,
                self.vLow - v,
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
                torch.mm(torch.neg(torch.transpose(torch.index_select(self.A, 1, torch.from_numpy(
                    np.arange(self.M - self.numSwitches, self.M)).long()), 0, 1)), vall.T).T +
                2 * (torch.mul((pij - pji), self.Rall) + torch.mul((qij - qji), self.Xall))[:, -self.numSwitches:None]
                - self.bigM * (1 - y),
                -torch.mm(torch.neg(torch.transpose(torch.index_select(self.A, 1, torch.from_numpy(
                    np.arange(self.M - self.numSwitches, self.M)).long()), 0, 1)), vall.T).T -
                2 * (torch.mul((pij - pji), self.Rall) + torch.mul((qij - qji), self.Xall))[:, -self.numSwitches:None]
                - self.bigM * (1 - y),
                1 - torch.mm(self.Aabs.double(), (zij + zji).T).T
            ], dim=1)
        else:
            # no bounds provided for the feeder(s)
            # Fixing the multiple feeder network design
            # if self.numFeeders > 1:
            #     # multiple feeders; located at end
            #     gen_ind = -1
            #     inds = np.arange(0, self.N-1)
            # else:
            #     # single feeder; located at beginning
            #     gen_ind = 0
            #     inds = np.arange(1, self.N)

            gen_ind = 0
            inds = np.arange(1, self.N)

            resids = torch.cat([
                pg[:, inds] - self.pgUpp[idx, :],
                self.pgLow[idx, :] - pg[:, inds],
                qg[:, inds] - self.qgUpp[idx, :],
                self.qgLow[idx, :] - qg[:, inds],
                - torch.reshape(plf, (-1, 1)),
                - torch.reshape(qlf, (-1, 1)),
                - torch.reshape(pg[:, gen_ind], (-1, 1)),
                - torch.reshape(qg[:, gen_ind], (-1, 1)),
                v - self.vUpp,
                self.vLow - v,
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
                torch.mm(torch.neg(torch.transpose(torch.index_select(self.A, 1, torch.from_numpy(
                    np.arange(self.M - self.numSwitches, self.M)).long()), 0, 1)), vall.T).T +
                2 * (torch.mul((pij - pji), self.Rall) + torch.mul((qij - qji), self.Xall))[:, -self.numSwitches:None]
                - self.bigM * (1 - y),
                -torch.mm(torch.neg(torch.transpose(torch.index_select(self.A, 1, torch.from_numpy(
                    np.arange(self.M - self.numSwitches, self.M)).long()), 0, 1)), vall.T).T -
                2 * (torch.mul((pij - pji), self.Rall) + torch.mul((qij - qji), self.Xall))[:, -self.numSwitches:None]
                - self.bigM * (1 - y),
                1 - torch.mm(self.Aabs.double(), (zij + zji).T).T
            ], dim=1)


        # - torch.mm(-torch.index_select(self.A, 0, torch.from_numpy(np.arange(self.M-self.numSwitches, self.M)).long()).double(), vall.T).T - 2*(
        #         torch.mul((pij - pji), self.Rall) + torch.mul((qij - qji), self.Xall))[:, -self.numSwitches:None] - self.bigM*(1-y),

        return torch.clamp(resids, 0)

    def ineq_partial_grad(self):
        return

    def ineq_grad(self):
        return

    def eq_grad(self):
        return

    def corr_steps(self, x, z, zc, idx):
        pos_resids = self.ineq_resid(z, zc, idx)
        delz = 2*torch.mm(pos_resids, self.dineq_dz_fullmat)
        delzc = torch.matmul(self.dzc_dz_mat, delz.T).T
        return delz, delzc

    def corr_steps_partial(self, x, z, zc, idx):
        pos_resids = self.ineq_resid(z, zc, idx)
        delz = 2*torch.mm(pos_resids, self.dineq_dz_partialmat)
        delzc = torch.matmul(self.dzc_dz_mat, delz.T).T
        return delz, delzc

    def mixed_sigmoid(self, x, nnout, idx):
        # apply sigmoids here to get pg and v, and integer-ify the topology selection

        # parameter tau controls sharpness of the sigmoid function applied to binary variables. larger = sharper
        tau = 5
        # if epoch < 300:
        #     tau = 5
        # # elif epoch < 800:
        # #     tau = 10
        # else:
        #     tau = 10

        z = mixed_sigmoid(self)(nnout, tau)

        # z = [zji, y\last, pij, pji, qji, {qij}sw, v\f, plf, qlf]  # initial guess from NN
        # zji and y\last are binary vars, which have already been processed to be between 0 and 1
        # process pij, pji, qji, {qij}_sw, v\f, plf, qlf to be within reasonable bounds
        zji, y_nol, pij_nn, pji_nn, qji_nn, qij_nosw_nn, v_nn, plf, qlf = self.decompose_vars_z(z)
        pij = pij_nn * self.bigM  # pij_min = 0; don't enforce any power flow constraints here
        pji = pji_nn * self.bigM  # pji_min = 0; don't enforce any power flow constraints here
        qji = qji_nn * self.bigM  # qji_min = 0; don't enforce any power flow constraints here
        qij_nosw = qij_nosw_nn * self.bigM
        v = v_nn*self.vUpp + (1-v_nn)*self.vLow
        z_physical = torch.hstack((zji, y_nol, pij, pji, qji, qij_nosw, v, plf.unsqueeze(1), qlf.unsqueeze(1)))

        return z_physical

    def output_layer(self, x, nnout, idx, design_code):
        # apply mixed sigmoid to zji var
        # apply random {0,1} selection to {y}\last
        # map other vars to physics w/ standard sigmoid

        # parameter tau controls sharpness of the sigmoid function applied to binary variables zji. larger = sharper
        tau = 5

        # z = mixedIntFnc(self)(nnout, tau)
        z = self.mixedIntOutput(nnout, tau, design_code)

        # z = [zji, y\last, pij, pji, qji, {qij}sw, v\f, plf, qlf]  # initial guess from NN
        # zji and y\last are binary vars, which have already been processed to be between 0 and 1
        # process pij, pji, qji, {qij}_sw, v\f, plf, qlf to be within physical bounds
        zji, y_nol, pij_nn, pji_nn, qji_nn, qij_nosw_nn, v_nn, plf, qlf = self.decompose_vars_z(z)
        pij = pij_nn * self.bigM  # pij_min = 0; don't enforce any power flow constraints here
        pji = pji_nn * self.bigM  # pji_min = 0; don't enforce any power flow constraints here
        qji = qji_nn * self.bigM  # qji_min = 0; don't enforce any power flow constraints here
        qij_nosw = qij_nosw_nn * self.bigM
        v = v_nn*self.vUpp + (1-v_nn)*self.vLow
        z_physical = torch.hstack((zji, y_nol, pij, pji, qji, qij_nosw, v, plf.unsqueeze(1), qlf.unsqueeze(1)))

        return z_physical

    def mixedIntOutput(self, z, tau, design_code):
        bin_vars_zji = z[:, 0:self.M]
        bin_vars_y = z[:, self.M:(self.M + self.numSwitches-1)]
        cont_vars = z[:, self.M+self.numSwitches-1:None]
        output_cont = nn.Sigmoid()(cont_vars)

        # modified sigmoid
        output_bin_zji = torch.clamp(2/(1+torch.exp(-tau*bin_vars_zji)) - 1, 0)

        batchsize = z.size(dim=0)
        # r = np.random.randint(-1, 1, batchsize)
        L_min = (self.N - 1) - (self.M - self.numSwitches)

        # output_bin_y = bin_vars_y.abs()
        # output_bin_y = nn.Sigmoid()(bin_vars_y) # cast between 0 and 1

        match design_code:
            case 1:
                output_bin_y = torch.clamp(bin_vars_y, 0, 1)
            case 2:
                output_bin_y = torch.sigmoid(bin_vars_y).clone()  # classic sigmoid - breaks backprop without clone()
            case 3:
                output_bin_y = torch.clamp(2/(1+torch.exp(-tau*bin_vars_y)) - 1, 0)  # modified sigmoid
            case _:
                # default action is to not do anything. design_codes 3 and 4 depend on epochs
                output_bin_y = bin_vars_y

        y_sorted_inds = torch.argsort(bin_vars_y)  # sorted in ascending order

        # output_bin_y = torch.clamp(bin_vars_y, 0, 1)
        # ceil the largest L values to 1, floor the smallest size(bin_y)-L values to 0
        rows_to_ceil = np.hstack((np.arange(0, batchsize).repeat(L_min))) # np.where(r == 0)[0]
        cols_to_ceil = np.hstack((y_sorted_inds[:, -L_min:].flatten())) # y_sorted_inds[np.where(r == 0)[0], -L_min-1]

        num_to_zero = bin_vars_y.size(dim=1) - L_min - 1
        if num_to_zero > 0:
            rows_to_floor = np.hstack((np.arange(0, batchsize).repeat(num_to_zero))) # np.where(r == -1)[0]
            cols_to_floor = np.hstack((y_sorted_inds[:, 0:num_to_zero].flatten())) # y_sorted_inds[np.where(r == -1)[0], -L_min-1]
        else:
            rows_to_floor = []
            cols_to_floor = []
        # output_bin_y[rows_to_ceil, cols_to_ceil] = output_bin_y[rows_to_ceil, cols_to_ceil].ceil()
        # output_bin_y[rows_to_floor, cols_to_floor] = output_bin_y[rows_to_floor, cols_to_floor].floor()
        output_bin_y[rows_to_ceil, cols_to_ceil] = torch.maximum(output_bin_y[rows_to_ceil, cols_to_ceil], torch.ones(1, len(output_bin_y[rows_to_ceil, cols_to_ceil])))  # 1
        output_bin_y[rows_to_floor, cols_to_floor] = torch.minimum(output_bin_y[rows_to_floor, cols_to_floor], torch.zeros(1, len(output_bin_y[rows_to_floor, cols_to_floor])))  # 0

        z_new = torch.hstack((output_bin_zji, output_bin_y, output_cont))
        return z_new

    def complete(self, x, z):  # , zslack
        z, zc = completion_function(self)(x, z)
        return z, zc


def completion_function(data):
    class completion_function_implemented(Function):
        @staticmethod
        def forward(ctx, x, z):
            # this will perform the completion step
            pl, ql = data.decompose_vars_x(x)
            zji, y_nol, pij, pji, qji, qij_sw, v, plf, qlf = data.decompose_vars_z(z)
            numsw = data.numSwitches
            ncases = x.shape[0]
            # zij + zji = 1
            # zij + zji = y
            # sum(y) = (N-1) - (M-numsw)
            y_rem = (data.N - 1) - (data.M - numsw) - torch.sum(y_nol, 1)
            zij = torch.zeros(ncases, data.M)
            zij[:, 0:data.M - numsw] = 1 - zji[:, 0:data.M - numsw]
            zij[:, data.M - numsw:-1] = y_nol - zji[:, data.M - numsw:-1]
            zij[:, -1] = y_rem - zji[:, -1]

            # Fixing the multiple feeder network design
            # if data.numFeeders > 1:  # multiple feeders (last few nodes). slack bus is last node
            #     # pg - pl = sum(pij - pji)
            #     pg = torch.hstack((pl, plf.unsqueeze(1))).T + torch.mm(data.A.double(),
            #                                                            torch.transpose(pij - pji, 0, 1))
            #     # vj - vi = -2(Rij(Pij-Pji) + Xij(Qij-Qji))
            #     vall = torch.hstack((v, torch.ones(ncases, 1)))
            # else:  # one feeder. slack bus is first node
            #     # pg - pl = sum(pij - pji)
            #     pg = torch.hstack((plf.unsqueeze(1), pl)).T + torch.mm(data.A.double(),
            #                                                            torch.transpose(pij - pji, 0, 1))
            #     # vj - vi = -2(Rij(Pij-Pji) + Xij(Qij-Qji))
            #     vall = torch.hstack((torch.ones(ncases, 1), v))

            # pg - pl = sum(pij - pji)
            pg = torch.hstack((plf.unsqueeze(1), pl)).T + torch.mm(data.A.double(),
                                                                   torch.transpose(pij - pji, 0, 1))
            # vj - vi = -2(Rij(Pij-Pji) + Xij(Qij-Qji))
            vall = torch.hstack((torch.ones(ncases, 1), v))

            delQ = torch.div((-0.5 * torch.matmul(-vall, data.A.double()) - torch.mul((pij-pji), data.Rall)), data.Xall)
            delQ_nosw = delQ[:, 0:-numsw]
            # qij = delQ + qji
            qij_rem = delQ_nosw + qji[:, 0:-numsw]
            qij = torch.hstack((qij_rem, qij_sw))
            # qg - ql = sum(qij - qji)

            delQ = torch.hstack((delQ_nosw, qij_sw - qji[:, data.M-numsw:]))
            # delQ = torch.hstack((delQ_nosw, qij_sw - qji[:, numsw:]))
            # Fixing the multiple feeder network design
            # if data.numFeeders > 1:  # multiple feeders (last few nodes). slack bus is last node
            #     qg = torch.hstack((ql, qlf.unsqueeze(1))).T + torch.mm(data.A.double(), torch.transpose(delQ, 0, 1))
            # else:  # one feeder. slack bus is first node
            #     qg = torch.hstack((qlf.unsqueeze(1), ql)).T + torch.mm(data.A.double(), torch.transpose(delQ, 0, 1))

            qg = torch.hstack((qlf.unsqueeze(1), ql)).T + torch.mm(data.A.double(), torch.transpose(delQ, 0, 1))

            zc = torch.hstack((zij, y_rem.unsqueeze(1), qij_rem, pg.T, qg.T))
            zc.requires_grad = True

            # # need to check the equality constraints....
            # pl, ql = data.decompose_vars_x(x)
            # _, _, pij_n, pji_n, qji_n, _, v_n, _, _ = data.decompose_vars_z(z)
            # _, _, qij_nosw_n, _, _ = data.decompose_vars_zc(zc)
            # qij_n = torch.hstack((qij_nosw_n, qij_sw))
            # vall_n = torch.hstack((torch.ones(ncases, 1), v_n))
            # res = torch.matmul(-vall_n, data.A.double()) + 2 * (
            #             torch.mul((pij_n - pji_n), data.Rall) + torch.mul((qij_n - qji_n), data.Xall))

            # print('specific constraint right after completion: {:.4f}'.format(torch.max(res)))

            # ctx.data = data
            return z, zc

        @staticmethod
        def backward(ctx, dl_dz, dl_dzc):
            # data = ctx.data
            dl_dz = dl_dz + torch.matmul(dl_dzc, data.dzc_dz)
            dl_dx = torch.matmul(dl_dzc, data.dzc_dx)

            return dl_dx, dl_dz

    return completion_function_implemented.apply

def mixed_sigmoid(data):
    class mixed_sigmoid_implemented(Function):
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

            bin_grad = grad_output[:, 0:data.M + data.numSwitches - 1].clone()  # what is clone() used for?
            cont_grad = grad_output[:, data.M + data.numSwitches - 1:None].clone()
            bin_vars = z[:, 0:data.M + data.numSwitches - 1]
            cont_vars = z[:, data.M + data.numSwitches - 1:None]

            # derivative of custom binary approx sigmoid
            bin_grad[bin_vars < 0] = 0  # for the first part of the function
            exp_res = torch.exp(-tau * bin_vars)
            inter_res = 2 * tau * exp_res / torch.square(1 + exp_res)
            bin_grad[bin_vars >= 0] *= inter_res[bin_vars >= 0]

            # bin_grad *= torch.exp(-bin_vars) / torch.square(1 + torch.exp(-bin_vars))

            # derivative of traditional sigmoid - do we need to do this explicitly?
            cont_grad *= torch.exp(-cont_vars) / torch.square(1 + torch.exp(-cont_vars))

            # grad_input = grad_output.clone()
            grad_input = torch.hstack((bin_grad, cont_grad))

            return grad_input, None

    return mixed_sigmoid_implemented.apply
#
#
# def mixedIntFnc(data):
#     class MixedIntOutput(Function):
#         # Output layer of NN
#         # zji: sigmoid approximation for indicator function, used for binary variable selection
#         # y\last: rounding
#         # other vars: standard sigmoid
#         # Reference sig approx: page 7 of https://arxiv.org/abs/2004.02402
#         @staticmethod
#         def forward(ctx, z, tau):
#             bin_vars_zji = z[:, 0:data.M]
#             bin_vars_y = z[:, data.M:(data.M + data.numSwitches-1)]
#             cont_vars = z[:, data.M+data.numSwitches-1:None]
#             output_cont = nn.Sigmoid()(cont_vars)
#             # output_bin = nn.Sigmoid()(bin_vars)
#
#             output_bin_zji = torch.clamp(2/(1+torch.exp(-tau*bin_vars_zji)) - 1, 0)
#
#             # # random y selection
#             # r = random.randint(-1, 0)  # np.random.randint(-1,0,batchsize)
#             # # swstats_on = (data.N - 1) - (data.M - data.numSwitches) - 1
#             # # swstats_off = data.numSwitches - swstats_on - 1
#             # # y_off = torch.zeros(batchsize, swstats_off)
#             # # y_on = torch.ones(batchsize, swstats_on)
#             # L_min = (data.N - 1) - (data.M - data.numSwitches)  # number of switches that are selected to be on (= 1)
#             # # L_max = L_min + r
#             # y_sorted_inds = np.argsort(bin_vars_y)  # sorted in ascending order
#             batchsize = z.size(dim=0)
#             r = np.random.randint(-1, 1, batchsize)
#             L_min = (data.N - 1) - (data.M - data.numSwitches)
#             y_sorted_inds = torch.argsort(bin_vars_y)  # sorted in ascending order
#             # value = [1]*(L_min*batchsize + np.size(np.where(r == 0)[0]))
#             # rows = np.hstack((np.arange(0, batchsize).repeat(L_min), np.where(r == 0)[0]))
#             # cols = np.hstack((y_sorted_inds[:, -L_min:].flatten(), y_sorted_inds[np.where(r == 0)[0], -L_min-1]))
#             # output_bin_y = torch.sparse_coo_tensor([rows, cols], value, (batchsize, data.numSwitches-1)).to_dense()  # inds, value, (size)
#
#             output_bin_y_test = bin_vars_y.abs()
#             # ceil the largest L values to 1, floor the smallest size(bin_y)-L values to 0
#             rows_to_ceil = np.hstack((np.arange(0, batchsize).repeat(L_min),
#                                       np.where(r == 0)[0]))
#             cols_to_ceil = np.hstack((y_sorted_inds[:, -L_min:].flatten(),
#                                       y_sorted_inds[np.where(r == 0)[0], -L_min-1]))
#
#             num_to_zero = bin_vars_y.size(dim=1) - L_min - 1
#             rows_to_floor = np.hstack((np.arange(0, batchsize).repeat(num_to_zero),
#                                        np.where(r == -1)[0]))
#             cols_to_floor = np.hstack((y_sorted_inds[:, 0:num_to_zero].flatten(),
#                                        y_sorted_inds[np.where(r == -1)[0], -L_min-1]))
#
#             output_bin_y_test[rows_to_ceil, cols_to_ceil] = output_bin_y_test[rows_to_ceil, cols_to_ceil].ceil()
#             output_bin_y_test[rows_to_floor, cols_to_floor] = output_bin_y_test[rows_to_floor, cols_to_floor].floor()
#
#             # output_bin_y = torch.zeros(batchsize, data.numSwitches-1) # initialize all at zeros
#             # for the L largest y values, set these to 1
#
#             z_new = torch.hstack((output_bin_zji, output_bin_y_test, output_cont))
#
#             ctx.save_for_backward(z_new)
#             ctx.data = data
#             ctx.tau = tau
#             return z_new
#
#         @staticmethod
#         def backward(ctx, grad_output):
#             z, = ctx.saved_tensors
#             data = ctx.data
#             tau = ctx.tau
#
#             bin_grad = grad_output[:, 0:data.M + data.numSwitches - 1].clone()  # what is clone() used for?
#             cont_grad = grad_output[:, data.M + data.numSwitches - 1:None].clone()
#             bin_vars = z[:, 0:data.M + data.numSwitches - 1]
#             cont_vars = z[:, data.M + data.numSwitches - 1:None]
#
#             # derivative of custom binary approx sigmoid
#             bin_grad[bin_vars < 0] = 0  # for the first part of the function
#             exp_res = torch.exp(-tau * bin_vars)
#             inter_res = 2 * tau * exp_res / torch.square(1 + exp_res)
#             bin_grad[bin_vars >= 0] *= inter_res[bin_vars >= 0]
#
#             # bin_grad *= torch.exp(-bin_vars) / torch.square(1 + torch.exp(-bin_vars))
#
#             # derivative of traditional sigmoid - do we need to do this explicitly?
#             cont_grad *= torch.exp(-cont_vars) / torch.square(1 + torch.exp(-cont_vars))
#
#             # grad_input = grad_output.clone()
#             grad_input = torch.hstack((bin_grad, cont_grad))
#
#             return grad_input, None
#
#     return MixedIntOutput.apply


# # datasets for training, validation, testing
# trainX, validX, textX
# obj_fn(Y), ineq_dist(X,Y), eq_resid(X,Y)
# ineq_partial_grad, ineq_grad, eq_grad
# xdim, ydim, nknowns, neq


# instantiate the class
# load in class data from file
# if file doesn't exist, load in original data then run setup. save class data
# have method for objective function and the gradients

# original data
# has network topology, limits, load and gen data
#