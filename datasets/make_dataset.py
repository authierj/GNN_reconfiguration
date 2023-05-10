import os
import pickle
import torch

import sys

sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from data_processing import OPTreconfigure

network_ID = "baranwu33"  # "node4"
# dataflag='partial'
dataflag = "full"

filename = 'casedata_33_uniform_extrasw4.mat'
# filename = "casedata_graph.mat"

torch.set_default_dtype(torch.float64)
filepath = os.path.join("datasets", network_ID, "raw", filename)
problem = OPTreconfigure(filepath, dataflag)

# Cut down number of samples if needed
num = 4000  # 5000  # 1200
problem._x = problem.x[:num, :]  # TODO: check dimensions and slicing
problem._num = problem.x.shape[0]
# if dataflag.lower() == "full":
#     problem._z = problem.y[:num, :]
#     problem._zc = problem.zc[:num, :]
#     problem._y = torch.hstack((problem._z, problem._zc))


with open(network_ID + "_dataset_test_2", "wb") as f:
    pickle.dump(problem, f)

# os.path.join('C:\','UserData','z004c27r','PycharmProjects','reconfiguration','datasets','acopf','matlab_datasets','FeasiblePairs_Case57.mat')
