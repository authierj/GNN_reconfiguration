# Extern
import torch
import numpy as np
import pickle
import argparse
from torch_geometric.loader import DataLoader

# Local
from utils_JA import Utils
from utils_JA import total_loss, dict_agg
from NN_layers import readout
from datasets.graphdataset import *
from NN_models.classical_gnn import GCN_Global_MLP_reduced_model, GCN_local_MLP
from NN_models.gated_switch import GatedSwitchGNN_globalMLP


def main(args):
    """
    main takes args as input, train and test a neural network and returns the directory where the results are stored

    args:
        args: dictionary with arguments
    return:
        save_dir: directory where the results are stored
    """
    # Making the code device-agnostic
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device)
    dataset_name = args["network"] + "_" + "dataset_test"
    filepath = "datasets/" + args["network"] + "/processed/" + dataset_name
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Network file does not exist")

    utils = Utils(data)
    graph_dataset = GraphDataSet(root="datasets/" + args["network"])
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

    num_epochs = args["epochs"]

    save_dir = os.path.join(
        "results",
        model.__class__.__name__,
        "_".join(
            [f'{args["numLayers"]}', f'{args["hiddenFeatures"]}', f'{args["lr"]:.0e}']
        ),
    )
    
    i = 0
    while os.path.exists(save_dir + f"_v{i}"):
        i += 1 
    save_dir = save_dir + f"_v{i}"
    os.makedirs(save_dir)

    stats = {}
    file = os.path.join(save_dir, "stats.dict")
    # train and test
    for i in range(num_epochs):

        if i == 100:
            print(i)

        train_epoch_stats = train(model, optimizer, cost_fnc, train_loader, args, utils)
        valid_epoch_stats = test_or_validate(model, cost_fnc, valid_loader, args, utils)

        print(
            f"Epoch: {i:03d}, Train Loss: {train_epoch_stats['train_loss']:.4f}, Valid Loss: {valid_epoch_stats['valid_loss']:.4f}"
        )

        if args["saveAllStats"]:
            #fmt: off
            if i == 0:
                for key in train_epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(train_epoch_stats[key]), axis=0)
                for key in valid_epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(valid_epoch_stats[key]), axis=0)
            #fmt: on
            else:
                for key in train_epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key],np.expand_dims(np.array(train_epoch_stats[key]), axis=0)))
                for key in valid_epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key],np.expand_dims(np.array(valid_epoch_stats[key]), axis=0)))
        else:
            stats = train_epoch_stats

        if i % args["resultsSaveFreq"] == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "model.dict"))
            with open(file, "wb") as f:
                # np.save(f, stats)
                pickle.dump(stats, f)

    with open(file, "wb") as f:
        pickle.dump(stats, f)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.dict"))

    return save_dir


def train(model, optimizer, criterion, loader, args, utils):
    """
    train the model for one epoch

    args:
        model: model to train
        optimizer: optimizer to use
        criterion: loss function
        loader: data loader
        args: dictionary of arguments
        utils: utils object
    return:
        epoch_stats: dictionary of statistics for the epoch
    """

    model.train()

    size = len(loader) * args["batchSize"]
    epoch_stats = {}

    # TODO change data structure to save
    for data in loader:
        z_hat, zc_hat = model(data, utils)
        train_loss, soft_weight = total_loss(
            z_hat,
            zc_hat,
            criterion,
            utils,
            args,
            data.idx,
            utils.A,
            train=True,
        )

        train_loss.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        dispatch_dist = utils.opt_dispatch_dist_JA(z_hat.detach(), zc_hat.detach(), data.y.detach())
        topology_dist = utils.opt_topology_dist_JA(z_hat.detach(), data.y.detach(), data.switch_mask.detach())
        # fmt: off
        dict_agg(epoch_stats, 'train_loss', torch.sum(train_loss).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'train_dispatch_error_max', torch.sum(torch.max(dispatch_dist, dim=1)[0]).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'train_dispatch_error_mean', torch.sum(torch.mean(dispatch_dist, dim=1)).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'train_dispatch_error_min', torch.sum(torch.min(dispatch_dist, dim=1)[0]).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'train_topology_error_max', torch.sum(torch.max(topology_dist, dim=1)[0]).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'train_topology_error_mean', torch.sum(torch.mean(topology_dist, dim=1)).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'train_topology_error_min', torch.sum(torch.min(topology_dist, dim=1)[0]).detach().cpu().numpy()/size, op='sum')
        # fmt: on
    return epoch_stats


def test_or_validate(model, criterion, loader, args, utils):
    """
    test the model on the test set or vallidation set

    args:
        model: model to test
        criterion: loss function
        loader: data loader
        args: dictionary of arguments
        utils: utils object
    return:
        epoch_stats: dictionary of statistics for the epoch
    """

    model.eval()
    size = len(loader) * args["batchSize"]
    epoch_stats = {}

    for data in loader:
        z_hat, zc_hat = model(data, utils)
        valid_loss, soft_weight = total_loss(
            z_hat,
            zc_hat,
            criterion,
            utils,
            args,
            data.idx,
            utils.A,
            train=True,
        )

        dispatch_dist = utils.opt_dispatch_dist_JA(z_hat.detach(), zc_hat.detach(), data.y.detach())
        topology_dist = utils.opt_topology_dist_JA(z_hat.detach(), data.y.detach(), data.switch_mask.detach())
        # fmt: off
        dict_agg(epoch_stats, 'valid_loss', torch.sum(valid_loss).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'valid_dispatch_error_max', torch.sum(torch.max(dispatch_dist, dim=1)[0]).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'valid_dispatch_error_mean', torch.sum(torch.mean(dispatch_dist, dim=1)).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'valid_dispatch_error_min', torch.sum(torch.min(dispatch_dist, dim=1)[0]).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'valid_topology_error_max', torch.sum(torch.max(topology_dist, dim=1)[0]).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'valid_topology_error_mean', torch.sum(torch.mean(topology_dist, dim=1)).detach().cpu().numpy()/size, op='sum')
        dict_agg(epoch_stats, 'valid_topology_error_min', torch.sum(torch.min(topology_dist, dim=1)[0]).detach().cpu().numpy()/size, op='sum')
        # fmt: on
    return epoch_stats


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
        "--epochs", type=int, default=1000, help="number of neural network epochs"
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
