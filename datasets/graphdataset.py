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
        if key in ["y", "switch_mask"]:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class GraphDataSet(InMemoryDataset):
    def __init__(self, root="datasets/node4"):
        super(GraphDataSet, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "casedata_33_uniform_extrasw"

    @property
    def processed_file_names(self):
        return "test_graph.pt"

    def process(self):
        # Read data into huge `Data` list.
        path = os.path.join(self.raw_dir, self.raw_file_names)
        data = spio.loadmat(path)

        cases = data["case_data_all"][0][0]
        network = data["network_33_data"][0, 0]
        solutions = data["res_data_all"][0][0]

        edges = from_np(network["bi_edge_index"] - 1).long()
        pl = cases["PL"]
        ql = cases["QL"]
        z = solutions["z"]
        zc = solutions["zc"]
        numSwitches = np.squeeze(network[8]).item(0)
        M = np.squeeze(network[6]).item(0)

        perm = np.arange(pl.shape[1])
        rng = np.random.default_rng(1)
        rng.shuffle(perm)

        pl = pl[:, perm].T
        ql = ql[:, perm].T
        z = z[:, perm].T
        zc = zc[:, perm].T

        x = torch.dstack((from_np(pl), from_np(ql))).float()
        y = torch.hstack((from_np(z), from_np(zc))).float()
        switch_mask = np.zeros(M)
        switch_mask[-numSwitches::] = 1
        switch_mask = from_np(switch_mask).bool()

        data_list = []

        for i in range(y.shape[0]):
            features = torch.cat((torch.zeros(1, x.shape[2]), x[i, :, :]), 0)
            graph = MyGraph(x=features, edge_index=edges, idx=i, y=y[i, :],switch_mask=switch_mask)
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SwitchGraph(Graph):
    "define Graphs with batches in a new dimension as in PyTorch"

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in [
            "x_mod",
            "A",
            "S",
            "Incidence",
            "Incidence_parent",
            "Incidence_child",
            "switch_mask",
            "D_inv",
            "y",
        ]:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class GraphDataSetWithSwitches(InMemoryDataset):
    def __init__(self, root="datasets/node4"):
        super(GraphDataSetWithSwitches, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "casedata_graph"
        # return "casedata_33_uniform_extrasw"


    @property
    def processed_file_names(self):
        return "graph_switches.pt"

    def process(self):
        # Read data into huge `Data` list.
        path = os.path.join(self.raw_dir, self.raw_file_names)
        data = spio.loadmat(path)

        cases = data["case_data_all"][0][0]
        network = data["network_4_data"][0, 0]
        # network = data["network_33_data"][0, 0]
        solutions = data["res_data_all"][0][0]

        # Process information about edges and switches
        edges = network["bi_edge_index"] - 1
        M = np.squeeze(network[6]).item(0)
        switch_indexes = np.squeeze((network["swInds"] - 1))
        numSwitches = np.squeeze(network[8]).item(0)


        interim = list(set(np.arange(0, M)) ^ set(switch_indexes))  # non-switch lines
    
        mStart = network[9]
        mEnd = network[10]

        mStart_reshaped = sp.hstack((mStart[:, interim], mStart[:, switch_indexes]))
        mEnd_reshaped = sp.hstack((mEnd[:, interim], mEnd[:, switch_indexes]))

        mStart_tensor = torch.sparse_coo_tensor(
            torch.vstack(
                (
                    torch.from_numpy(mStart_reshaped.tocoo().row),
                    torch.from_numpy(mStart_reshaped.tocoo().col),
                )
            ),
            mStart_reshaped.data,
            torch.Size(mStart_reshaped.shape),
        ).to_dense().bool()  # indices, values, size
        mEnd_tensor = torch.sparse_coo_tensor(
            torch.vstack(
                (
                    torch.from_numpy(mEnd_reshaped.tocoo().row),
                    torch.from_numpy(mEnd_reshaped.tocoo().col),
                )
            ),
            mEnd_reshaped.data,
            torch.Size(mEnd_reshaped.shape),
        ).to_dense().bool()  # indices, values, size
        
        incidence = mStart_tensor.int() - mEnd_tensor.int()
        D_inv = torch.diag(1/torch.sum(torch.abs(incidence), 1))

        switch_mask = np.zeros(M)
        switch_mask[-numSwitches::] = 1
        switch_indexes_bi = np.concatenate((switch_indexes, switch_indexes + M))
        switches = edges[:, switch_indexes_bi]
        edges_without_switches = np.delete(edges, switch_indexes_bi, axis=1)

        # Adjacency and Switch-Adjacency matrices
        N = np.squeeze(network["N"])
        A = np.zeros((N, N))
        S = np.zeros((N, N))

        A[edges_without_switches[0, :], edges_without_switches[1, :]] = 1
        S[switches[0, :], switches[1, :]] = 1

        # Convert to torch
        switches = from_np(switches).long()
        edges_without_switches = from_np(edges_without_switches).long()
        A = from_np(A).bool()
        A.to_sparse()
        S = from_np(S).bool()
        S.to_sparse()
        switch_mask = from_np(switch_mask).bool()

        pl = cases["PL"]
        ql = cases["QL"]
        z = solutions["z"]
        zc = solutions["zc"]

        # I think this is useless
        perm = np.arange(pl.shape[1])
        rng = np.random.default_rng(1)
        rng.shuffle(perm)

        pl = pl[:, perm].T
        ql = ql[:, perm].T
        z = z[:, perm].T
        zc = zc[:, perm].T

        x = torch.dstack((from_np(pl), from_np(ql))).float()
        y = torch.hstack((from_np(z), from_np(zc))).float()

        data_list = []

        for i in range(y.shape[0]):
            features = torch.cat((torch.zeros(1, x.shape[2]), x[i, :, :]), 0)
            # graph = MyGraph(
            #     x=features,
            #     edge_index=edges_without_switches,
            #     switch_index=switches,
            #     A=A,
            #     S=S,
            #     idx=i,
            #     y=y[i, :],
            # )
            graph = SwitchGraph(
                x_mod=features,
                A=A,
                S=S,
                switch_mask=switch_mask,
                Incidence=incidence,
                Incidence_parent=mStart_tensor,
                Incidence_child=mEnd_tensor,
                D_inv=D_inv,
                idx=i,
                y=y[i, :],
            )
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = GraphDataSetWithSwitches(root="datasets/node4")
