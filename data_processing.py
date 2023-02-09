import torch
import numpy as np
import scipy.io as spio
import scipy.sparse as sp
from setup import *

class OPTreconfigure():
    def __init__(self, filename, dataflag='partial', valid_frac=0.1, test_frac=0.1):
        #TODO look at this class, it's a mess
        
        # load network data by name... - replace this function
        data = spio.loadmat(filename)

        cases = data['case_data_all'][0][0]
        
        ########################## Data Extraction #############################
        pl = cases[0]
        ql = cases[1]
        pgLow = cases[2]
        pgUpp = cases[3]
        qgLow = cases[4]
        qgUpp = cases[5]

        network = data['network_33_data'][0, 0]  # network_4_data for node4 test case
        SBase = np.squeeze(network[0])
        VBase = np.squeeze(network[1])
        ZBase = np.squeeze(network[2])
        YBase = np.squeeze(network[3])
        IBase = np.squeeze(network[4])
        N = np.squeeze(network[5])
        M = np.squeeze(network[6])
        swInds = np.squeeze(network[7])
        numSwitches = np.squeeze(network[8])
        mStart = network[9]
        mEnd = network[10]
        Rall = np.squeeze(network[13])
        Xall = np.squeeze(network[14])
        vLow = np.square(np.squeeze(network[15]))
        vUpp = np.square(np.squeeze(network[16]))
        bigM = 0.5  # all quantities in pu, so 1pu = SBase

        # network data
        self.N = N.item(0)
        self.M = M.item(0)
        self.SBase = SBase.item(0)

        self.swInds = swInds  # doesn't need to be a tensor bc an index
        self.numSwitches = numSwitches.item(0)


        ######################## Data processing ###############################
        interim = list(set(np.arange(0, M)) ^ set(self.swInds - 1))  # non-switch lines
        self.Rall = torch.cat((torch.from_numpy(Rall[interim]),
                                torch.from_numpy(Rall[swInds - 1])))
        self.Xall = torch.cat((torch.from_numpy(Xall[interim]),
                                torch.from_numpy(Xall[swInds - 1])))
        
        mStart_reshaped = sp.hstack((mStart[:, interim], mStart[:, swInds - 1]))
        mEnd_reshaped = sp.hstack((mEnd[:, interim], mEnd[:, swInds - 1]))
        Acoo = mStart_reshaped - mEnd_reshaped

        mStart_tensor = torch.sparse_coo_tensor(
            torch.vstack((torch.from_numpy(mStart_reshaped.tocoo().row),
            torch.from_numpy(mStart_reshaped.tocoo().col))),
            mStart_reshaped.data, torch.Size(mStart_reshaped.shape))  # indices, values, size
        mEnd_tensor = torch.sparse_coo_tensor(
            torch.vstack((torch.from_numpy(mEnd_reshaped.tocoo().row),
            torch.from_numpy(mEnd_reshaped.tocoo().col))),
            mEnd_reshaped.data, torch.Size(mEnd_reshaped.shape))  # indices, values, size

        self.mStart = mStart_tensor
        self.mEnd = mEnd_tensor
        self.A = (mStart_tensor - mEnd_tensor).to_dense()
        self.Aabs = mStart_tensor + mEnd_tensor
        
        # Pytorch want each row a case, each column a node
        # randomize the data so different cases appear in training, validation, testing
        perm = np.arange(pl.shape[1])
        np.random.shuffle(perm)  # in place
        # specific load + generation data
        self.pl = torch.from_numpy(pl[:, perm].T)  # pl and ql given for all nodes except feeder
        self.ql = torch.from_numpy(ql[:, perm].T)
        # limits do NOT include feeder
        self.pgUpp = torch.from_numpy(pgUpp[:, perm].T)
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

        # this is all the data - then it'll get split into train-test-valid
        # TODO fix dimensions for multiple test cases
        x = np.concatenate([pl, ql])  # each row is a node, each column a test case
        if dataflag.lower() == 'full':
            # TODO LOAD SOLUTION DATA
            solutions = data['res_data_all'][0][0]
            # z, zc, yalmiptime, solvertime, objVal
            # pg, qg, v, yij, zij, zji, pij, pji, qij, qji = get_solution_data()
            # z = np.concatenate([zji, yij[0:-1], pij, pji, qji, qij[-numSwitches:None], v[1:None], [pl[0, :]], [ql[0, :]]])
            # zc = np.concatenate([zij, [yij[-1, :]], qij[0:M-numSwitches], pg, qg])
            z = solutions[0]
            zc = solutions[1]
            objvals = solutions[3]  # this was solutions[4] for node_4 because changes were made later.
            # TODO: fix node_4 test case to match the new code

            self._z = torch.from_numpy(z[:, perm].T)
            self._zc = torch.from_numpy(zc[:, perm].T)
            self._y = torch.hstack((self._z, self._zc))
            self._objval = torch.from_numpy(objvals[:, perm].T)
        else:
            self._z = None
            self._zc = None
            self._y = None
            self._objval = None

        self._x = torch.t(torch.tensor(x, dtype=torch.get_default_dtype()))
        self._known = torch.t(torch.tensor(1, dtype=torch.get_default_dtype()))

        self._xdim = 2*(N-1)  # x
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
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def zdim(self):
        return self._zdim

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
        res = self.y()
        if res is not None:
            return res[:int(self.num * self.train_frac)]
        else:
            return None

    @property
    def validY(self):
        res = self.y()
        if res is not None:
            return res[int(self.num * self.train_frac):int(self.num * (self.train_frac + self.valid_frac))]
        else:
            return None

    @property
    def testY(self):
        res = self.y()
        if res is not None:
            return res[int(self.num * (self.train_frac + self.valid_frac)):]
        else:
            return None

    @property
    def device(self):
        return self._device

    def train_adjustIDX(self, idx):
        start, _ = self.train_idx
        return start + idx

    def test_adjustIDX(self, idx):
        start, _ = self.test_idx
        return start + idx

    def valid_adjustIDX(self, idx):
        start, _ = self.valid_idx
        return start + idx

    def decompose_vars_x(self, x):
        # x = [pl\f, ql\f]  # input to NN
        pl = x[:, 0:self.N - 1]
        ql = x[:, self.N - 1:None]
        return pl, ql