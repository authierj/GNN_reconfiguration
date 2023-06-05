import torch
import os
import scipy.io as spio
import numpy as np
import scipy.sparse as sp
from torch import from_numpy as from_np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as Graph


class MyGraph(Graph):
    "define Graphs with batches in a new dimension as in PyTorch"

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["y", "qg_upp", "pg_upp", "x_data"]:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class GraphDataSet(InMemoryDataset):
    def __init__(self, root="datasets/node4"):
        super(GraphDataSet, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "casedata_33_uniform_extrasw4_new_costfnc"
        # return "casedata_33_uniform_extrasw"

    @property
    def processed_file_names(self):
        return "test_newcostfnc_nosw.pt"

    def extract_JA_sol(self, z, zc, n, m, numSwitches, sw_idx, no_sw_idx):
        y_nol = z[:, m + np.arange(0, numSwitches - 1)]
        pij = z[:, m + numSwitches - 1 + np.arange(0, m)]
        pji = z[:, 2 * m + numSwitches - 1 + np.arange(0, m)]
        qji = z[:, 3 * m + numSwitches - 1 + np.arange(0, m)]
        qij_sw = z[:, 4 * m + numSwitches - 1 + np.arange(0, numSwitches)]
        v = z[
            :, 4 * m + 2 * numSwitches - 1 + np.arange(0, n - 1)
        ]  # feeder not included
        v = np.hstack((np.ones((v.shape[0], 1)), v))
        plf = z[:, 4 * m + 2 * numSwitches - 1 + n - 1]
        qlf = z[:, 4 * m + 2 * numSwitches - 1 + n]

        ylast = np.expand_dims(zc[:, m], axis=1)
        qij_nosw = zc[:, m + 1 + np.arange(0, m - numSwitches)]
        pg = zc[:, 2 * m + 1 - numSwitches + np.arange(0, n)]
        qg = zc[:, 2 * m + 1 - numSwitches + n + np.arange(0, n)]

        qij = np.hstack((qij_nosw, qij_sw))

        p_flow = pij - pji
        q_flow = qij - qji
        p_flow_reshaped = np.hstack((p_flow[:, no_sw_idx], p_flow[:, sw_idx]))
        q_flow_reshaped = np.hstack((q_flow[:, no_sw_idx], q_flow[:, sw_idx]))

        pg[:, 0] = pg[:, 0] - plf
        qg[:, 0] = qg[:, 0] - qlf
        topo = np.hstack((np.ones((ylast.shape[0], m - numSwitches)), y_nol, ylast))

        z_JA = np.hstack((p_flow_reshaped, v, topo))
        zc_JA = np.hstack((q_flow_reshaped, pg, qg))

        return z_JA, zc_JA

    def process(self):
        # Read data into huge `Data` list.
        path = os.path.join(self.raw_dir, self.raw_file_names)
        data = spio.loadmat(path)

        cases = data["case_data_all"][0][0]
        network = data["network_data"][0, 0]
        solutions = data["res_data_all"][0][0]

        pl = cases["PL"].T
        ql = cases["QL"].T
        x = torch.dstack((from_np(pl), from_np(ql)))

        mean_x = torch.mean(x, dim=0)
        std_x = torch.std(x, dim=0)
        std_x[0,:] = 1
        norm_x = (x - mean_x) / std_x
        
        pg_upp = from_np(cases["PGUpp"].T)
        qg_upp = from_np(cases["QGUpp"].T)

        n = np.squeeze(network[5]).item(0)
        m = np.squeeze(network[6]).item(0)

        sw_idx = np.squeeze(network[8]) - 1
        interim = list(set(np.arange(0, m)) ^ set(sw_idx))
        numSwitches = np.squeeze(network[9]).item(0)

        begin_edges = network[10].indices
        end_edges = network[11].indices
        edges = np.vstack((begin_edges, end_edges))
        edges = np.hstack((edges[:, interim], edges[:, sw_idx]))
        reversed_edges = np.flip(edges, axis=0)
        bi_edges = from_np(np.hstack((edges, reversed_edges))).long()

        z = solutions[0]
        zc = solutions[1]

        z_JA, zc_JA = self.extract_JA_sol(z.T, zc.T, n, m, numSwitches, sw_idx, interim)

        y = torch.hstack((from_np(z_JA), from_np(zc_JA)))

        data_list = []

        for i in range(y.shape[0]):
            graph = MyGraph(
                x=norm_x[i, :, :],
                edge_index=bi_edges,
                x_data=x[i, :, :],
                pg_upp=pg_upp[i, :],
                qg_upp=qg_upp[i, :],
                idx=i,
                y=y[i, :],
            )
            data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class GraphDataSetWithSwitches(InMemoryDataset):
    def __init__(self, root="datasets/node4"):
        super(GraphDataSetWithSwitches, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "casedata_33_uniform_extrasw4_new_costfnc"
        # return "casedata_33_uniform_extrasw"

    @property
    def processed_file_names(self):
        return "test_newcostfnc.pt"

    def extract_JA_sol(self, z, zc, n, m, numSwitches, sw_idx, no_sw_idx):
        y_nol = z[:, m + np.arange(0, numSwitches - 1)]
        pij = z[:, m + numSwitches - 1 + np.arange(0, m)]
        pji = z[:, 2 * m + numSwitches - 1 + np.arange(0, m)]
        qji = z[:, 3 * m + numSwitches - 1 + np.arange(0, m)]
        qij_sw = z[:, 4 * m + numSwitches - 1 + np.arange(0, numSwitches)]
        v = z[
            :, 4 * m + 2 * numSwitches - 1 + np.arange(0, n - 1)
        ]  # feeder not included
        v = np.hstack((np.ones((v.shape[0], 1)), v))
        plf = z[:, 4 * m + 2 * numSwitches - 1 + n - 1]
        qlf = z[:, 4 * m + 2 * numSwitches - 1 + n]

        ylast = np.expand_dims(zc[:, m], axis=1)
        qij_nosw = zc[:, m + 1 + np.arange(0, m - numSwitches)]
        pg = zc[:, 2 * m + 1 - numSwitches + np.arange(0, n)]
        qg = zc[:, 2 * m + 1 - numSwitches + n + np.arange(0, n)]

        qij = np.hstack((qij_nosw, qij_sw))

        p_flow = pij - pji
        q_flow = qij - qji
        p_flow_reshaped = np.hstack((p_flow[:, no_sw_idx], p_flow[:, sw_idx]))
        q_flow_reshaped = np.hstack((q_flow[:, no_sw_idx], q_flow[:, sw_idx]))

        pg[:, 0] = pg[:, 0] - plf
        qg[:, 0] = qg[:, 0] - qlf
        topo = np.hstack((np.ones((ylast.shape[0], m - numSwitches)), y_nol, ylast))

        z_JA = np.hstack((p_flow_reshaped, v, topo))
        zc_JA = np.hstack((q_flow_reshaped, pg, qg))

        return z_JA, zc_JA

    def process(self):
        # Read data into huge `Data` list.
        path = os.path.join(self.raw_dir, self.raw_file_names)
        data = spio.loadmat(path)

        cases = data["case_data_all"][0][0]
        network = data["network_data"][0, 0]
        solutions = data["res_data_all"][0][0]

        pl = cases["PL"].T
        ql = cases["QL"].T
        x = torch.dstack((from_np(pl), from_np(ql)))

        mean_x = torch.mean(x, dim=0)
        std_x = torch.std(x, dim=0)
        std_x[0,:] = 1
        norm_x = (x - mean_x) / std_x
        
        pg_upp = from_np(cases["PGUpp"].T)
        qg_upp = from_np(cases["QGUpp"].T)

        n = np.squeeze(network[5]).item(0)
        m = np.squeeze(network[6]).item(0)

        sw_idx = np.squeeze(network[8]) - 1
        interim = list(set(np.arange(0, m)) ^ set(sw_idx))
        numSwitches = np.squeeze(network[9]).item(0)

        begin_edges = network[10].indices
        end_edges = network[11].indices
        edges = np.vstack((begin_edges, end_edges))

        edges_no_sw = edges[:, interim]
        reversed_edges_no_sw = np.flip(edges_no_sw, axis=0)
        bi_edges_no_sw = from_np(np.hstack((edges_no_sw, reversed_edges_no_sw))).long()

        edges_sw = edges[:, sw_idx]
        reversed_edges_sw = np.flip(edges_sw, axis=0)
        bi_edges_sw = from_np(np.hstack((edges_sw, reversed_edges_sw))).long()

        A = torch.sparse_coo_tensor(
            bi_edges_no_sw, torch.ones(bi_edges_no_sw.shape[1]), (n, n)
        )
        S = torch.sparse_coo_tensor(
            bi_edges_sw, torch.ones(bi_edges_sw.shape[1]), (n, n)
        )

        z = solutions[0]
        zc = solutions[1]

        z_JA, zc_JA = self.extract_JA_sol(z.T, zc.T, n, m, numSwitches, sw_idx, interim)

        y = torch.hstack((from_np(z_JA), from_np(zc_JA)))

        data_list = []

        for i in range(y.shape[0]):
            graph = MyGraph(
                x=norm_x[i, :, :],
                x_data=x[i, :, :],
                A=A.to_dense().bool(),
                S=S.to_dense().bool(),
                pg_upp=pg_upp[i, :],
                qg_upp=qg_upp[i, :],
                idx=i,
                y=y[i, :],
            )
            data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
