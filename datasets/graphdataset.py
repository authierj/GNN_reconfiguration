import torch
import os
import scipy.io as spio
import numpy as np
from torch import from_numpy as from_np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as Graph


class GraphDataSet(InMemoryDataset):

    def __init__(self, root='datasets/node4'):
        super(GraphDataSet, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'casedata_graph'

    @property
    def processed_file_names(self):
        return 'test_graph.pt'

    def process(self):
        # Read data into huge `Data` list.
        path = os.path.join(self.raw_dir, self.raw_file_names)
        data = spio.loadmat(path)

        cases = data['case_data_all'][0][0]
        network = data['network_4_data'][0, 0]
        solutions = data['res_data_all'][0][0]

        edges = from_np(network['bi_edge_index']-1).long()
        pl = cases['PL']
        ql = cases['QL']
        z = solutions['z']
        zc = solutions['zc']

        perm = np.arange(pl.shape[1])
        np.random.shuffle(perm)

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


dataset = GraphDataSet(root='datasets/node4')
print("HI()")
