import torch
import os
import scipy.io as spio
import numpy as np
from torch import from_numpy as from_np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as Graph


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
            graph = Graph(x=features, edge_index=edges, idx=i, y=y[i, :])
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyGraph(Graph):
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

    @property
    def processed_file_names(self):
        return "graph_switches.pt"

    def process(self):
        # Read data into huge `Data` list.
        path = os.path.join(self.raw_dir, self.raw_file_names)
        data = spio.loadmat(path)

        cases = data["case_data_all"][0][0]
        network = data["network_4_data"][0, 0]
        solutions = data["res_data_all"][0][0]
        incidence = from_np(network["Aflow"].todense()).int()
        incidence_parent = from_np(network["mStart"].todense()).bool()
        incidence_child = from_np(network["mEnd"].todense()).bool()
        D_inv = torch.diag(1/torch.sum(torch.abs(incidence), 0))
        Incidence = incidence

        # Process information about edges and switches
        edges = network["bi_edge_index"] - 1
        num_edges = int(edges.shape[1] / 2)
        # indicates which edges are switches
        switch_indexes = np.squeeze((network["swInds"] - 1))
        switch_mask = np.zeros(num_edges)
        switch_mask[switch_indexes] = 1
        switch_indexes_bi = np.concatenate((switch_indexes, switch_indexes + num_edges))
        switches = edges[:, switch_indexes_bi]
        edges_without_switches = np.delete(edges, switch_indexes_bi, axis=1)
        # R_vector = np.squeeze(network["R"])
        # X_vector = np.squeeze(network["X"])
        # Adjacency and Switch-Adjacency matrices
        N = np.squeeze(network["N"])
        A = np.zeros((N, N))
        S = np.zeros((N, N))
        # R = np.zeros((N, N))
        # X = np.zeros((N, N))
        A[edges_without_switches[0, :], edges_without_switches[1, :]] = 1
        S[switches[0, :], switches[1, :]] = 1
        # R[edges[0, :], edges[1, :]] = np.hstack((R_vector, R_vector))
        # X[edges[0, :], edges[1, :]] = np.hstack((X_vector, X_vector))

        # Convert to torch
        switches = from_np(switches).long()
        edges_without_switches = from_np(edges_without_switches).long()
        A = from_np(A).bool()
        A.to_sparse()
        S = from_np(S).bool()
        S.to_sparse()
        switch_mask = from_np(switch_mask).bool()
        # R = from_np(R).float()
        # R.to_sparse()
        # X = from_np(X).float()
        # X.to_sparse()

        pl = cases["PL"]
        ql = cases["QL"]
        z = solutions["z"]
        zc = solutions["zc"]

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
            graph = MyGraph(
                x_mod=features,
                A=A,
                S=S,
                switch_mask=switch_mask,
                Incidence=Incidence,
                Incidence_parent=incidence_parent,
                Incidence_child=incidence_child,
                D_inv=D_inv,
                idx=i,
                y=y[i, :],
            )
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = GraphDataSetWithSwitches(root="datasets/node4")
print("HI()")
