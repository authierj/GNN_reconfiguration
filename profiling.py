# Extern
import torch
import numpy as np
import pickle
import argparse
from torch_geometric.loader import DataLoader
import time
import torch.autograd.profiler as profiler

# Local
from utils_JA import Utils
from utils_JA import total_loss, dict_agg
from NN_layers import readout
from datasets.graphdataset import *
from NN_models.classical_gnn import GCN_Global_MLP_reduced_model, GCN_local_MLP
from NN_models.gated_switch import GatedSwitchGNN_globalMLP


def main(args):
   
    # Making the code device-agnostic
    if torch.cuda.is_available():
        print(torch.cuda.current_device())
        x = torch.rand(5, 3).cuda()
        print(x.device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args["device"] = device
    print("Using device: ", device)
    dataset_name = args["network"] + "_" + "dataset_test"
    # filepath = "datasets/" + args["network"] + "/processed/" + dataset_name
    filepath = os.path.join("datasets", args["network"], "processed", dataset_name)

    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Network file does not exist")

    utils = Utils(data, device)
    graph_dataset = GraphDataSet(root="datasets/" + args["network"])
    graph_dataset.data.to(device)
    # graph_dataset = graph_dataset.to_device(device)
    # graph_dataset = GraphDataSetWithSwitches(root="datasets/" + args["network"])

    # TODO change to arguments so that we can use different networks directly
    train_graphs = graph_dataset[0:3200]
    valid_graphs = graph_dataset[3200:3600]
    test_graphs = graph_dataset[3600:4000]

    batch_size = args["batchSize"]
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=True)

    # Model initialization and optimizer
    output_dim = utils.M + utils.N + utils.numSwitches

    # model = GatedSwitchGNN_globalMLP(args, utils.N, output_dim)
    # model = GCN_Global_MLP_reduced_model(args, utils.N, output_dim)
    model = GCN_local_MLP(args, utils.N, output_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=5e-4)
    cost_fnc = utils.obj_fnc_JA


    for data in train_loader:
        model(data, utils)
        model.train()
        with profiler.profile(with_stack=True, profile_memory=True) as prof:
            z_hat, zc_hat = model(data, utils)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

        break   



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        default="baranwu33",
        choices=["node4", "IEEE13", "baranwu33"],
        help="network identification",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of neural network epochs"
    )
    parser.add_argument(
        "--batchSize", type=int, default=200, help="training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="neural network learning rate"
    )
    parser.add_argument(
        "--numLayers", type=int, default=4, help="the number of layers in the GNN"
    )
    parser.add_argument(
        "--inputFeatures",
        type=int,
        default=2,
        help="number of features in the input layer of the GNN",
    )
    parser.add_argument(
        "--hiddenFeatures",
        type=int,
        default=4,
        help="number of features in the hidden layers of the GNN",
    )
    parser.add_argument(
        "--softWeight",
        type=float,
        default=100,
        help="total weight given to constraint violations in loss",
    )
    parser.add_argument(
        "--useAdaptiveWeight",
        type=bool,
        default=False,
        help="whether constraint violation weight is time-varying",
    )
    parser.add_argument(
        "--useVectorWeight",
        type=bool,
        default=False,
        help="whether constraint violation weight is vector",
    )
    parser.add_argument(
        "--adaptiveWeightLr",
        type=float,
        default=1e-2,
        help="constraint violation adaptive weight learning rate",
    )
    parser.add_argument(
        "--useCompl", type=bool, default=True, help="whether to use completion"
    )
    parser.add_argument(
        "--useTrainCorr",
        type=bool,
        default=False,
        help="whether to use correction during training",
    )
    parser.add_argument(
        "--useTestCorr",
        type=bool,
        default=False,
        help="whether to use correction during testing",
    )
    parser.add_argument(
        "--corrTrainSteps",
        type=int,
        default=5,
        help="number of correction steps during training",
    )
    parser.add_argument(
        "--corrTestMaxSteps",
        type=int,
        default=5,
        help="max number of correction steps during testing",
    )
    parser.add_argument(
        "--corrEps", type=float, default=1e-3, help="correction procedure tolerance"
    )
    parser.add_argument(
        "--corrLr",
        type=float,
        default=1e-4,
        help="learning rate for correction procedure",
    )
    parser.add_argument(
        "--corrMomentum",
        type=float,
        default=0.5,
        help="momentum for correction procedure",
    )
    parser.add_argument(
        "--saveAllStats",
        type=bool,
        default=True,
        help="whether to save all stats, or just those from latest epoch",
    )
    parser.add_argument(
        "--resultsSaveFreq",
        type=int,
        default=50,
        help="how frequently (in terms of number of epochs) to save stats to file",
    )
    parser.add_argument(
        "--saveModel",
        type=bool,
        default=True,
        help="determine if the trained model will be saved",
    )
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument(
        "--aggregation", type=str, default="max", choices=["sum", "mean", "max"]
    )
    parser.add_argument(
        "--norm", type=str, default="batch", choices=["batch", "layer", "none"]
    )
    parser.add_argument("--gated", type=bool, default=True)

    args = parser.parse_args()
    main(vars(args))



