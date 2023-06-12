# Extern
import torch
import numpy as np
import pickle
import argparse
from torch_geometric.loader import DataLoader, DenseDataLoader
import time
import os

# Local
from utils_JA import Utils
from utils_JA import total_loss, dict_agg
from NN_layers import readout
from datasets.graphdataset import GraphDataSetWithSwitches
import NN_models.gated_switch as gated_gnn


def main(args):
    """
    main takes args as input, train and test a neural network and returns the directory where the results are stored

    args:
        args: dictionary with arguments
    return:
        save_dir: directory where the results are stored
    """
    print(args)
    total_time_start = time.time()
    # Making the code device-agnostic
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args["device"] = device
    print("Using device: ", device)
    dataset_name = args["network"] + "_" + "extrasw4_endswitch"
    filepath = os.path.join("datasets", args["network"], "processed", dataset_name)

    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Network file does not exist")

    utils = Utils(data, device)
    graph_dataset = GraphDataSetWithSwitches(root="datasets/" + args["network"])
    graph_dataset.data.to(device)

    # TODO change to 7800, 400, 400 
    train_graphs = graph_dataset[0:7800]
    valid_graphs = graph_dataset[7800:8200]
    test_graphs = graph_dataset[8200:8600]

    batch_size = args["batchSize"]
    train_loader = DenseDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    valid_loader = DenseDataLoader(valid_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DenseDataLoader(test_graphs, batch_size=batch_size, shuffle=True)

    model = getattr(gated_gnn, args["model"])(args, utils)
    # model = gated_gnn.simple_MLP(args, utils)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=5e-4)
    cost_fnc = utils.obj_fnc_JA

    num_epochs = args["epochs"]

    if args["topoLoss"]:
        save_dir = os.path.join(
            "results",
            "basic_experiments_topoLoss",
            model.__class__.__name__,
            "_".join(
                [
                    f'{args["numLayers"]}',
                    f'{args["hiddenFeatures"]}',
                    f'{args["lr"]:.0e}',
                ]
            ),
        )
    else:
        save_dir = os.path.join(
            "results",
            "new_topo",
            model.__class__.__name__,
            "_".join(
                [
                    f'{args["numLayers"]}',
                    f'{args["hiddenFeatures"]}',
                    f'{args["lr"]:.0e}',
                ]
            ),
        )
    if args["saveModel"] or args["saveAllStats"]:
        i = 0
        while os.path.exists(os.path.join(save_dir, f"v{i}")):
            i += 1
            if i >= 10:
                print("experiment already ran 10 times")
                return save_dir
        save_dir = os.path.join(save_dir, f"v{i}")
        os.makedirs(save_dir)
        file = os.path.join(save_dir, "stats.dict")

    stats = {}
    warm_start = False
    # train and test
    for i in range(num_epochs):
        if i == 0 and args["warmStart"]:
            warm_start = True
        if i == 999:
            print("hi")
        start_train = time.time()
        train_epoch_stats = train(
            model, optimizer, cost_fnc, train_loader, args, utils, warm_start
        )
        end_train = time.time()
        train_time = end_train - start_train
        valid_epoch_stats = test_or_validate(
            model, cost_fnc, valid_loader, args, utils, warm_start
        )

        if i % 10 == 0:
            print(
                f"Epoch: {i:03d}, Train Loss: {train_epoch_stats['train_loss']:.4f}, Valid Loss: {valid_epoch_stats['valid_loss']:.4f}, Train Time: {train_time:.4f}"
            )

        if args["saveAllStats"]:
            # fmt: off
            if i == 0:
                for key in train_epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(train_epoch_stats[key]), axis=0)
                for key in valid_epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(valid_epoch_stats[key]), axis=0)
            # fmt: on
            else:
                for key in train_epoch_stats.keys():
                    stats[key] = np.concatenate(
                        (
                            stats[key],
                            np.expand_dims(np.array(train_epoch_stats[key]), axis=0),
                        )
                    )
                for key in valid_epoch_stats.keys():
                    stats[key] = np.concatenate(
                        (
                            stats[key],
                            np.expand_dims(np.array(valid_epoch_stats[key]), axis=0),
                        )
                    )
        else:
            stats = train_epoch_stats

        if i % args["resultsSaveFreq"] == 0 and (
            args["saveModel"] or args["saveAllStats"]
        ):
            torch.save(model.state_dict(), os.path.join(save_dir, "model.dict"))
            with open(file, "wb") as f:
                # np.save(f, stats)
                pickle.dump(stats, f)

    if args["saveModel"] or args["saveAllStats"]:
        with open(file, "wb") as f:
            pickle.dump(stats, f)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.dict"))

    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print(f"Total time: {total_time:.4f}")

    return save_dir


def train(model, optimizer, criterion, loader, args, utils, warm_start=False):
    """
    train the model for one epoch

    args:
        model: model to train
        optimizer: optimizer to use
        criterion: loss function
        loader: data loader
        args: dictionary of arguments
        utils: utils object
        warm_start: whether to warm start the PhyR
    return:
        epoch_stats: dictionary of statistics for the epoch
    """

    size = len(loader) * args["batchSize"]
    epoch_stats = {}
    
    model.train()
    for data in loader:
        z_hat, zc_hat = model(data, utils, warm_start)
        train_loss = total_loss(
            z_hat,
            zc_hat,
            criterion,
            utils,
            args,
            data.pg_upp,
            data.qg_upp,
            utils.A,
            data.y
        )
        if args["topoLoss"]:
            train_loss += args["topoWeight"] * utils.squared_error_topology(
                z_hat, data.y
            )
            # train_loss += args["topoWeight"] * utils.cross_entropy_loss_topology(
            #     z_hat, data.y
            # )
        if args["pushProb"]:
            train_loss += args["pushWeight"] * utils.prob_push(z_hat)

        train_loss.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        
        dispatch_dist = utils.opt_dispatch_dist_JA(zc_hat.detach(), data.y.detach())
        voltage_dist = utils.opt_voltage_dist_JA(z_hat.detach(), data.y.detach())
        ineq_resid = utils.ineq_resid_JA(
            z_hat.detach(), zc_hat.detach(), data.pg_upp, data.qg_upp, utils.A
        )
        topology_dist = utils.opt_topology_dist_JA(z_hat.detach(), data.y.detach())
        topo_factor = utils.M / utils.numSwitches
        eps_converge = args["corrEps"]
        # fmt: off
        dict_agg(epoch_stats, 'train_loss', torch.sum(train_loss).detach().cpu().numpy()/size, op='sum')
        if args["saveModel"]:
            dict_agg(epoch_stats, 'train_ineq_max', torch.sum(torch.max(ineq_resid, dim=1)[0]).detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'train_ineq_mean', torch.sum(torch.mean(ineq_resid, dim=1)).detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'train_ineq_min', torch.sum(torch.min(ineq_resid, dim=1)[0]).detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'train_ineq_num_viol_0', torch.sum(ineq_resid > eps_converge).float().detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'train_ineq_num_viol_1', torch.sum(ineq_resid > 10 * eps_converge).float().detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'train_ineq_num_viol_2', torch.sum(ineq_resid > 100 * eps_converge).float().detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'train_dispatch_error_max', torch.max(torch.mean(dispatch_dist, dim=1)).detach().cpu().numpy()/len(loader), op='sum')
            dict_agg(epoch_stats, 'train_dispatch_error_mean', torch.sum(torch.mean(dispatch_dist, dim=1)).detach().cpu().numpy()/size, op='sum')
            dict_agg(epoch_stats, 'train_dispatch_error_min', torch.min(torch.mean(dispatch_dist, dim=1)).detach().cpu().numpy()/len(loader), op='sum')
            dict_agg(epoch_stats, 'train_voltage_error_mean', torch.sum(torch.mean(voltage_dist, dim=1)).detach().cpu().numpy()/size, op='sum')
            dict_agg(epoch_stats, 'train_topology_error_max', torch.max(torch.mean(topo_factor*topology_dist, dim=1)).detach().cpu().numpy()/len(loader), op='sum')
            dict_agg(epoch_stats, 'train_topology_error_mean', torch.sum(torch.mean(topo_factor*topology_dist, dim=1)).detach().cpu().numpy()/size, op='sum')
            dict_agg(epoch_stats, 'train_topology_error_min', torch.min(torch.mean(topo_factor*topology_dist, dim=1)).detach().cpu().numpy()/len(loader), op='sum')
        # fmt: on
    return epoch_stats


def test_or_validate(model, criterion, loader, args, utils, warm_start=False):
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

    i = 0
    for data in loader:
        z_hat, zc_hat = model(data, utils, warm_start)
        valid_loss = total_loss(
            z_hat,
            zc_hat,
            criterion,
            utils,
            args,
            data.pg_upp,
            data.qg_upp,
            utils.A,
            data.y
        )

        dispatch_dist = utils.opt_dispatch_dist_JA(zc_hat.detach(), data.y.detach())
        voltage_dist = utils.opt_voltage_dist_JA(z_hat.detach(), data.y.detach())
        ineq_resid = utils.ineq_resid_JA(
            z_hat.detach(), zc_hat.detach(), data.pg_upp, data.qg_upp, utils.A
        )
        topology_dist = utils.opt_topology_dist_JA(z_hat.detach(), data.y.detach())
        topo_factor = utils.M / utils.numSwitches
        # opt_valid_loss = total_loss(
        #     data.y[:,:utils.zrdim],
        #     data.y[:,utils.zrdim:],
        #     criterion,
        #     utils,
        #     args,
        #     data.pg_upp,
        #     data.qg_upp,
        #     utils.A,
        #     data.y
        # )
        # opt_gap = (valid_loss - opt_valid_loss)/opt_valid_loss      
        eps_converge = args["corrEps"]
        utils.optimality_distance(z_hat.detach(), zc_hat.detach(), data.y.detach())
        
        dict_agg(
            epoch_stats,
            "valid_loss",
            torch.sum(valid_loss).detach().cpu().numpy() / size,
            op="sum",
        )
        if args["saveModel"]:
            # fmt: off
            dict_agg(epoch_stats, 'valid_ineq_max', torch.sum(torch.max(ineq_resid, dim=1)[0]).detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'valid_ineq_mean', torch.sum(torch.mean(ineq_resid, dim=1)).detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'valid_ineq_min', torch.sum(torch.min(ineq_resid, dim=1)[0]).detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'valid_ineq_num_viol_0', torch.sum(ineq_resid > eps_converge).float().detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'valid_ineq_num_viol_1', torch.sum(ineq_resid > 10 * eps_converge).float().detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'valid_ineq_num_viol_2', torch.sum(ineq_resid > 100 * eps_converge).float().detach().cpu().numpy()/size, op="sum")
            dict_agg(epoch_stats, 'valid_dispatch_error_max', torch.max(torch.mean(dispatch_dist, dim=1)).detach().cpu().numpy()/len(loader), op='sum')
            dict_agg(epoch_stats, 'valid_dispatch_error_mean', torch.sum(torch.mean(dispatch_dist, dim=1)).detach().cpu().numpy()/size, op='sum')
            dict_agg(epoch_stats, 'valid_dispatch_error_min', torch.min(torch.mean(dispatch_dist, dim=1)).detach().cpu().numpy()/len(loader), op='sum')
            dict_agg(epoch_stats, 'valid_voltage_error_mean', torch.sum(torch.mean(voltage_dist, dim=1)).detach().cpu().numpy()/size, op='sum')
            dict_agg(epoch_stats, 'valid_topology_error_max', torch.max(torch.mean(topo_factor*topology_dist, dim=1)).detach().cpu().numpy()/len(loader), op='sum')
            dict_agg(epoch_stats, 'valid_topology_error_mean', torch.sum(torch.mean(topo_factor*topology_dist, dim=1)).detach().cpu().numpy()/size, op='sum')
            dict_agg(epoch_stats, 'valid_topology_error_min', torch.min(torch.mean(topo_factor*topology_dist, dim=1)).detach().cpu().numpy()/len(loader), op='sum')
            # dict_agg(epoch_stats, 'valid_opt_gap', torch.mean(opt_gap).detach().cpu().numpy()/len(loader), op='sum')
            # fmt: on
        i += 1
    return epoch_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="GatedSwitchGNN",
        choices=[
            "GatedSwitchGNN",
            "GatedSwitchGNN_globalMLP",
        ],
        help="model to train",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="baranwu33",
        choices=["node4", "IEEE13", "baranwu33"],
        help="network identification",
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="number of neural network epochs"
    )
    parser.add_argument(
        "--batchSize", type=int, default=200, help="training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="neural network learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.1)
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
    parser.add_argument("--saveModel", action="store_true")
    parser.add_argument(
        "--saveAllStats",
        action="store_true",
        help="whether to save all stats, or just those from latest epoch",
    )
    parser.add_argument(
        "--resultsSaveFreq",
        type=int,
        default=50,
        help="how frequently (in terms of number of epochs) to save stats to file",
    )
    parser.add_argument(
        "--topoLoss", action="store_true", help="whether to use topology loss"
    )
    parser.add_argument(
        "--topoWeight", type=float, default=100, help="topology loss weight"
    )
    parser.add_argument(
        "--aggregation", type=str, default="max", choices=["sum", "mean", "max"]
    )
    parser.add_argument(
        "--norm", type=str, default="batch", choices=["batch", "layer", "none"]
    )
    parser.add_argument("--gated", type=bool, default=True)
    parser.add_argument(
        "--corrEps", type=float, default=1e-3, help="correction procedure tolerance"
    )
    parser.add_argument(
        "--switchActivation",
        type=str,
        default="sig",
        choices=["sig", "mod_sig", "None"],
    )
    parser.add_argument(
        "--warmStart", action="store_true", help="whether to warm start the PhyR"
    )
    parser.add_argument(
        "--PhyR",
        type=str,
        default="mod_PhyR",
        choices=["PhyR", "back_PhyR", "mod_PhyR", "mod_back_PhyR"],
    )
    parser.add_argument(
        "--pushProb", action="store_true", help="whether to use soft PhyR"
    )
    parser.add_argument(
        "--pushWeight",
        type=float,
        default=100,
        help="total weight given to soft PhyR",
    )

    args = parser.parse_args()
    main(vars(args))
