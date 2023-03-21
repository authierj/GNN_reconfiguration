# Extern
import torch
import numpy as np
import pickle
import argparse
from torch_geometric.loader import DataLoader

# Local
from utils_JA import Utils
from utils_JA import total_loss
from NN_layers import readout
from datasets.graphdataset import *
from plots import *
from NN_models.gated_switch import GatedSwitchGNN


def main(args):

    dataset_name = args["network"] + "_" + "dataset_test"
    filepath = "datasets/" + args["network"] + "/processed/" + dataset_name
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Network file does not exist")

    utils = Utils(data)
    graph_dataset = GraphDataSetWithSwitches(root="datasets/" + args["network"])

    # TODO change to arguments so that we can use different networks directly
    train_graphs = graph_dataset[0:3200]
    valid_graphs = graph_dataset[3200:3600]
    test_graphs = graph_dataset[3600:4000]

    batch_size = args["batchSize"]
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=True)

    # Model initialization and optimizer
    model = GatedSwitchGNN(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=5e-4)
    cost_fnc = utils.obj_fnc_JA

    num_epochs = args["epochs"]
    train_losses = np.zeros(num_epochs)
    valid_losses = np.zeros(num_epochs)

    # train and test
    for i in range(num_epochs):
        if i == 100:
            print(i)

        train_losses[i] = train(model, optimizer, cost_fnc, train_loader, args, utils)
        valid_losses[i] = test_or_validate(model, cost_fnc, valid_loader, args, utils)

        print(
            f"Epoch: {i:03d}, Train Loss: {train_losses[i]:.4f}, Valid Loss: {valid_losses[i]:.4f}"
        )

    if args["saveModel"]:
        description = "_".join((args["network"], args["GNN"], args["readout"]))
        torch.save(model.state_dict, os.path.join("trained_nn", description))

    return train_losses, valid_losses


def train(model, optimizer, criterion, loader, args, utils):
    model.train()

    epoch_loss = 0
    for data in loader:
        z_hat, zc_hat, x_input = model(data, utils)

        # Znew_train, ZCnew_train = grad_steps(
        #     x_input, z_hat, zc_hat, args, utils, data.idx, plotFlag=False
        # )
        train_loss, soft_weight = total_loss(
            x_input,
            z_hat,
            zc_hat,
            criterion,
            utils,
            args,
            data.idx,
            data.Incidence,
            train=True,
        )

        train_loss.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += train_loss.detach().mean()

    return epoch_loss / len(loader)


def test_or_validate(model, criterion, loader, args, utils):
    model.eval()

    test_loss_total = 0
    for data in loader:

        with torch.no_grad():
            z_hat, zc_hat, x_input = model(data, utils)

        # Znew_train, ZCnew_train = grad_steps(
        #     x_input, z_hat, zc_hat, args, utils, data.idx, plotFlag=False
        # )
        test_loss, soft_weight = total_loss(
            x_input,
            z_hat,
            zc_hat,
            criterion,
            utils,
            args,
            data.idx,
            data.Incidence,
            train=False,
        )
        test_loss_total += test_loss.detach().mean()

    return test_loss_total / len(loader)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        default="node4",
        choices=["node4", "IEEE13", "baranwu33"],
        help="network identification",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="number of neural network epochs"
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
        default=False,
        help="determine if the trained model will be saved",
    )
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--aggregation", type=str, default="sum", choices=["sum", "mean", "max"]
    )
    parser.add_argument(
        "--norm", type=str, default="batch", choices=["batch", "layer", "none"]
    )
    parser.add_argument("--gated", type=bool, default=True)

    args = parser.parse_args()
    main(vars(args))
