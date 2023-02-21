"trying to implement a simple GNN"

# Extern
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import argparse

from torch_geometric.loader import DataLoader

# Local
import NN_layers.graph_NN as graph_extraction
from utils_JA import Utils
from utils_JA import xgraph_xflatten
import NN_layers.readout as readout_layer
from datasets.graphdataset import GraphDataSet
from plots import *


def main(args):
    dataset_name = args['network'] + '_' + 'dataset'
    filepath = 'datasets/' + args['network'] + '/processed/' + dataset_name

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print('Network file does not exist')

    utils = Utils(data)

    graph_dataset = GraphDataSet(root='datasets/node4')
    train_graphs = graph_dataset[0:3200]
    valid_graphs = graph_dataset[3200:3600]
    test_graphs = graph_dataset[3600:4000]

    train_loader = DataLoader(
        train_graphs, batch_size=args['batchSize'], shuffle=True)
    valid_loader = DataLoader(
        valid_graphs, batch_size=len(valid_graphs), shuffle=True)
    test_loader = DataLoader(
        test_graphs, batch_size=len(test_graphs), shuffle=True)

    output_features_GNN = args['outputFeatures']
    dropout = args['dropout']

    GNN = getattr(graph_extraction, args['GNN'])(input_features=2, hidden_channels=args['hiddenFeatures'],
                                                 output_classes=output_features_GNN, layers=args['numLayers'], dropout=dropout)

    readout = getattr(readout_layer, args['readout'])(input_features=output_features_GNN*data.M,
                                                      hidden_channels=output_features_GNN*data.M,
                                                      output_classes=data.zdim, layers=2, dropout=dropout)

    model = MyEnsemble(GNN, readout, completion_step=args['useCompl'])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=5e-4)
    cost_fnc = utils.obj_fnc

    num_epochs = args['epochs']
    train_losses = np.zeros(num_epochs)
    valid_losses = np.zeros(num_epochs)
    for i in range(num_epochs):
        train_losses[i] = train(
            model, optimizer, cost_fnc, train_loader, args, utils)
        valid_losses[i] = test_or_validate(
            model, cost_fnc, valid_loader, args, utils)

        print(
            f'Epoch: {i:03d}, Train Loss: {train_losses[i]:.4f}, Valid Loss: {valid_losses[i]:.4f}')

    if args['saveModel']:
        description = '_'.join(args['network'], args['GNN'], args['readout'])
        torch.save(model.state_dict, os.path.join('trained_nn', description))

    loss_cruve(train_losses, valid_losses, args)

def train(model, optimizer, criterion, loader, args, utils):
    model.train()

    epoch_loss = 0
    for data in loader:
        z_hat, zc_hat = model(data.x, data.edge_index,
                              training=True, utils=utils)
        x_nn = xgraph_xflatten(data.x, args['batchSize'])
        # z_hat, zc_hat = utils.complete(x_nn, z)

        Znew_train, ZCnew_train = grad_steps(x_nn, z_hat, zc_hat, args, utils,
                                             data.idx, plotFlag=False)
        train_loss, soft_weight = total_loss(x_nn, Znew_train, ZCnew_train,
                                             criterion, utils, args, data.idx,
                                             train=True)

        train_loss.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += train_loss.sum()

    return epoch_loss/len(loader)


def test_or_validate(model, criterion, loader, args, utils):
    model.eval()

    test_loss_total = 0
    for data in loader:
        z_hat, zc_hat = model(data.x, data.edge_index,
                              training=False, utils=utils)
        x_nn = xgraph_xflatten(data.x, 400)
        # z_hat, zc_hat = utils.complete(x_nn, z)

        Znew_train, ZCnew_train = grad_steps(x_nn, z_hat, zc_hat, args, utils,
                                             data.idx, plotFlag=False)
        test_loss, soft_weight = total_loss(x_nn, Znew_train, ZCnew_train,
                                            criterion, utils, args, data.idx,
                                            train=False)
        test_loss_total += test_loss.sum()

    return test_loss_total/(2*len(loader))


def grad_steps(x, z, zc, args, utils, idx, plotFlag):
    """ 
    absolutely no clue about what happens here
    """

    take_grad_steps = args['useTrainCorr']
    # plotFlag = True
    if take_grad_steps:
        lr = args['corrLr']
        num_steps = args['corrTrainSteps']
        momentum = args['corrMomentum']

        z_new = z
        zc_new = zc
        old_delz = 0
        old_delphiz = 0

        z_new_c = z
        zc_new_c = zc
        old_delz_c = 0
        old_delphiz_c = 0

        if plotFlag:
            num_steps = 200  # 10000
            ineq_costs = np.array(())
            ineq_costs_c = np.array(())

        iters = 0
        for i in range(num_steps):

            delz, delphiz = utils.corr_steps(x, z_new, zc_new, idx)
            new_delz = lr * delz + momentum * old_delz
            new_delphiz = lr * delphiz + momentum * old_delphiz
            z_new = z_new - new_delz
            # FEB 14: something in correction not working. use this until debugged
            _, zc_new_compl = utils.complete(x, z_new)
            zc_new = torch.stack(list(zc_new_compl), dim=0)
            old_delz = new_delz
            old_delphiz = new_delphiz

            delz_c, delphiz_c = utils.corr_steps(x, z_new_c, zc_new_c, idx)
            new_delz_c = lr * delz_c + momentum * old_delz_c
            new_delphiz_c = lr * delphiz_c + momentum * old_delphiz_c
            z_new_c = z_new_c - new_delz_c
            _, zc_new_compl = utils.complete(x, z_new_c)
            zc_new_c = torch.stack(list(zc_new_compl), dim=0)
            old_delz_c = new_delz_c
            old_delphiz_c = new_delphiz_c

            check_z = torch.max(z_new - z_new_c)
            check_zc = torch.max(zc_new - zc_new_c)

            eq_check = utils.eq_resid(x, z_new, zc_new)
            eq_check_c = utils.eq_resid(x, z_new_c, zc_new_c)

            if torch.max(eq_check) > 1e-6:
                print('eq_update z broken'.format(torch.max(eq_check)))
            if torch.max(eq_check_c) > 1e-6:
                print('eq_update z compl broken'.format(torch.max(eq_check_c)))
            if check_z > 1e-6:
                print('z updates not consistent'.format(check_z))
            if check_zc > 1e-6:
                print('zc updates not consistent'.format(check_zc))

            if plotFlag:
                ineq_cost = torch.norm(utils.ineq_resid(
                    z_new.detach(), zc_new.detach(), idx), dim=1)
                ineq_costs = np.hstack(
                    (ineq_costs, ineq_cost[0].detach().numpy()))
                ineq_cost_c = torch.norm(
                    utils.ineq_resid(z_new_c, zc_new_c, idx), dim=1)
                ineq_costs_c = np.hstack(
                    (ineq_costs_c, ineq_cost_c[0].detach().numpy()))

            iters += 1

        if plotFlag:
            fig, ax = plt.subplots()
            fig_c, ax_c = plt.subplots()
            s = np.arange(1, iters + 1, 1)
            ax.plot(s, ineq_costs)
            ax_c.plot(s, ineq_costs_c)

        return z_new, zc_new
    else:
        return z, zc


def total_loss(x, z, zc, criterion, utils, args, idx, train):

    obj_cost = criterion(z, zc)
    ineq_dist = utils.ineq_resid(z, zc, idx)  # gives update for vector weight
    ineq_cost = torch.norm(ineq_dist, dim=1)  # gives norm for scalar weight

    soft_weight = args['softWeight']

    if train and args['useAdaptiveWeight']:
        if args['useVectorWeight']:
            soft_weight_new = (soft_weight
                               + args['adaptiveWeightLr']*ineq_dist.detach())
            return obj_cost + torch.sum((soft_weight
                                        + args['adaptiveWeightLr']*ineq_dist)
                                        * ineq_dist, 1), soft_weight_new
        else:
            soft_weight_new = soft_weight + \
                args['adaptiveWeightLr']*ineq_cost.detach()
            return obj_cost + ((soft_weight + args['adaptiveWeightLr']*ineq_cost)
                               * ineq_cost), soft_weight_new

    return obj_cost + soft_weight*ineq_cost, soft_weight


class MyEnsemble(nn.Module):
    def __init__(self, GNN, readout, completion_step):
        super().__init__()
        self.GNN = GNN
        self.readout = readout
        self.completion_step = completion_step

    def forward(self, x, graph, training, utils):
        xg = self.GNN(x, graph)

        if training:
            x_input = xgraph_xflatten(x, 200, first_node=False)
            x_nn = xgraph_xflatten(xg, 200, first_node=True)
        else:
            x_input = xgraph_xflatten(x, 400, first_node=False)
            x_nn = xgraph_xflatten(xg, 400, first_node=True)

        out = self.readout(x_nn)
        z = utils.output_layer(out)

        # TODO process x to have correct format to be sent to usual functions
        if self.completion_step:
            z, zc = utils.complete(x_input, z)
            zc_tensor = torch.stack(list(zc), dim=0)
            return z, zc_tensor

        else:
            return utils.process_output(x_input, out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='node4',
                        choices=['node4', 'IEEE13', 'baranwu33'], help='network identification')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of neural network epochs')
    parser.add_argument('--batchSize', type=int, default=200,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='neural network learning rate')
    parser.add_argument('--GNN', type=str, default='GCN',
                        help='model for feature extraction layers')
    parser.add_argument('--readout', type=str, default='GlobalMLP',
                        help='model for readout layer')
    parser.add_argument('--numLayers', type=int, default=2,
                        help='the number of layers in the GNN')
    parser.add_argument('--hiddenFeatures', type=int, default=4,
                        help='number of features in the hidden layers of the GNN')
    parser.add_argument('--outputFeatures', type=int, default=4,
                        help='number of features in the last layer of the GNN')
    parser.add_argument('--softWeight', type=float, default=100,
                        help='total weight given to constraint violations in loss')
    parser.add_argument('--useAdaptiveWeight', type=bool, default=False,
                        help='whether constraint violation weight is time-varying')
    parser.add_argument('--useVectorWeight', type=bool, default=False,
                        help='whether constraint violation weight is vector')
    parser.add_argument('--adaptiveWeightLr', type=float, default=1e-2,
                        help='constraint violation adaptive weight learning rate')
    parser.add_argument('--useCompl', type=bool, default=True,
                        help='whether to use completion')
    parser.add_argument('--useTrainCorr', type=bool, default=False,
                        help='whether to use correction during training')
    parser.add_argument('--useTestCorr', type=bool, default=False,
                        help='whether to use correction during testing')
    parser.add_argument('--corrTrainSteps', type=int, default=5,
                        help='number of correction steps during training')
    parser.add_argument('--corrTestMaxSteps', type=int, default=5,
                        help='max number of correction steps during testing')
    parser.add_argument('--corrEps', type=float, default=1e-3,
                        help='correction procedure tolerance')
    parser.add_argument('--corrLr', type=float, default=1e-4,
                        help='learning rate for correction procedure')
    parser.add_argument('--corrMomentum', type=float, default=0.5,
                        help='momentum for correction procedure')
    parser.add_argument('--saveAllStats', type=bool, default=True,
                        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int, default=50,
                        help='how frequently (in terms of number of epochs) to save stats to file')
    parser.add_argument('--saveModel', type=bool, default=False,
                        help='determine if the trained model will be saved')
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    main(vars(args))
