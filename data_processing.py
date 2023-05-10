import torch
import numpy as np
import scipy.io as spio
import scipy.sparse as sp
from torch_geometric.data import Data


class OPTreconfigure:
    def __init__(self, filename, dataflag="partial", valid_frac=0.1, test_frac=0.1):
        # load network data by name... - replace this function
        data = spio.loadmat(filename)

        cases = data["case_data_all"][0][0]
        ########################## Data Extraction #############################
        pl = cases[0]
        ql = cases[1]
        x = np.concatenate([pl, ql])
        pgLow = cases[2]
        pgUpp = cases[3]
        qgLow = cases[4]
        qgUpp = cases[5]

        network = data["network_data"][0, 0]
        # network = data["network_4_data"][0, 0]

        SBase = np.squeeze(network[0])
        N = np.squeeze(network[5])
        M = np.squeeze(network[6])
        swInds = np.squeeze(network[8])-1
        numSwitches = np.squeeze(network[9])
        mStart = network[10]
        mEnd = network[11]
        edges = np.vstack((mStart.indices, mEnd.indices))
        Rall = np.squeeze(network[14])
        Xall = np.squeeze(network[15])
        vLow = np.square(np.squeeze(network[16]))
        vUpp = np.square(np.squeeze(network[17]))
        bigM = 0.5  # all quantities in pu, so 1pu = SBase

        # network data
        self.N = N.item(0)
        self.M = M.item(0)
        self.SBase = SBase.item(0)

        self.swInds = swInds  # doesn't need to be a tensor bc an index
        self.numSwitches = numSwitches.item(0)

        ######################## Data processing ###############################
        interim = list(set(np.arange(0, M)) ^ set(self.swInds))  # non-switch lines
        self.Rall = torch.cat(
            (torch.from_numpy(Rall[interim]), torch.from_numpy(Rall[swInds]))
        )
        self.Xall = torch.cat(
            (torch.from_numpy(Xall[interim]), torch.from_numpy(Xall[swInds]))
        )

        mStart_reshaped = sp.hstack((mStart[:, interim], mStart[:, swInds]))
        mEnd_reshaped = sp.hstack((mEnd[:, interim], mEnd[:, swInds]))
        edges_reshaped = np.hstack((edges[:, interim], edges[:, swInds]))
        Acoo = mStart_reshaped - mEnd_reshaped

        mStart_tensor = torch.sparse_coo_tensor(
            torch.vstack(
                (
                    torch.from_numpy(mStart_reshaped.tocoo().row),
                    torch.from_numpy(mStart_reshaped.tocoo().col),
                )
            ),
            mStart_reshaped.data,
            torch.Size(mStart_reshaped.shape),
        )  # indices, values, size
        mEnd_tensor = torch.sparse_coo_tensor(
            torch.vstack(
                (
                    torch.from_numpy(mEnd_reshaped.tocoo().row),
                    torch.from_numpy(mEnd_reshaped.tocoo().col),
                )
            ),
            mEnd_reshaped.data,
            torch.Size(mEnd_reshaped.shape),
        )  # indices, values, size

        self.mStart = mStart_tensor
        self.mEnd = mEnd_tensor
        self.A = (mStart_tensor - mEnd_tensor).to_dense()

        self.D_inv = torch.diag(1 / torch.sum(torch.abs(self.A), 1))

        edges_without_switches = edges_reshaped[:, :-self.numSwitches]
        switches = edges_reshaped[:, -self.numSwitches:]

        # Adjacency and Switch-Adjacency matrices
        Adj = np.zeros((N, N))
        S = np.zeros((N, N))

        Adj[edges_without_switches[0, :], edges_without_switches[1, :]] = 1
        Adj[edges_without_switches[1, :], edges_without_switches[0, :]] = 1

        S[switches[0, :], switches[1, :]] = 1
        S[switches[1, :], switches[0, :]] = 1

        # Convert to torch
        Adj = torch.from_numpy(Adj).bool()
        self.Adj = Adj
        S = torch.from_numpy(S).bool()
        self.S = S

        # specific load + generation data
        # pl and ql given for all nodes except feeder
        self.pl = torch.from_numpy(pl.T)
        self.ql = torch.from_numpy(ql.T)
        self._x = torch.from_numpy(x.T)
        # limits do NOT include feeder
        self.pgUpp = torch.from_numpy(pgUpp.T)
        self.pgLow = torch.from_numpy(pgLow.T)
        self.qgUpp = torch.from_numpy(qgUpp.T)
        self.qgLow = torch.from_numpy(qgLow.T)
        self.vUpp = torch.tensor(vUpp, dtype=torch.get_default_dtype())  # scalar value
        self.vLow = torch.tensor(vLow, dtype=torch.get_default_dtype())  # scalar value
        self.bigM = bigM

        self._xdim = 2 * (N - 1)  # x
        self._zdim = N + M + numSwitches  # only z
        self._zcdim = M + 2 * N  # only zc
        self._ydim = self._zdim + self._zcdim  # z, zc
        self._num = self._x.shape[0]  # number of test cases
        self._neq = 2 * M + 1 - numSwitches + 2 * N  # equalities
        # M+1+N+N+M-#sw+M = 2*M + 1 + 2*N - numSwitches
        self._nineq = 6 * (N - 1) + 4 + 10 * M + 3 * numSwitches + N  # inequalities
        self._nknowns = 1  # number of known vars (and not x) vfeed

        # Define train/valid/test split
        self._valid_frac = valid_frac
        self._test_frac = test_frac

    def __str__(self):
        return "NETWORK-{}-{}-{}".format(self.N, self.valid_frac, self.test_frac)

    @property
    def x(self):
        return self._x

    # @property
    # def y(self):
    #     return self._y

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
        res = self._y
        if res is not None:
            return res[: int(self.num * self.train_frac)]
        else:
            return None

    @property
    def validY(self):
        res = self._y
        if res is not None:
            return res[
                int(self.num * self.train_frac) : int(
                    self.num * (self.train_frac + self.valid_frac)
                )
            ]
        else:
            return None

    @property
    def testY(self):
        res = self._y
        if res is not None:
            return res[int(self.num * (self.train_frac + self.valid_frac)) :]
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
        pl = x[:, 0 : self.N - 1]
        ql = x[:, self.N - 1 : None]
        return pl, ql

