"trying to implement a simple GNN"

# Extern
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# Local
from graph_NN import GCN
from utils_JA import Utils
from utils_JA import xgraph_xflatten
from readout import GlobalMLP
import default_args


def main():
    print("Hello")

    filepath = 'node4_dataset'
    batch_size = 200
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print('Network file does not exist')

    args = default_args.method_default_args()
    utils = Utils(data)

    # Once we use bigger examples, we should use the Dataset Class of GPY
    dataset_train, dataset_valid, dataset_test = create_graph_datasets(data)

    train_loader = DataLoader(dataset_train,
                              batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, len(dataset_valid), shuffle=True)
    test_loader = DataLoader(dataset_test, len(dataset_test), shuffle=True)

    hidden_features_GNN = 12
    output_features_GNN = 12
    GNN = GCN(input_features=2, hidden_channels=hidden_features_GNN,
              output_classes=output_features_GNN, layers=4)

    hidden_features_MLP = 12
    readout = GlobalMLP(input_features=output_features_GNN*data.M, hidden_channels=hidden_features_MLP*data.M,
                        output_classes=data.zdim, layers=2, dropout=0.5)
    print(readout)
    model = MyEnsemble(GNN, readout)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    cost_fnc = utils.obj_fnc

    for epoch in range(1, 100):
        loss = train(model, optimizer, cost_fnc, train_loader, args, utils)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        eval_loss = test_or_validate(model, cost_fnc, valid_loader, args, utils)
        print(f'Eval_Loss: {eval_loss:.4f}')

    # TODO test here


def create_graph_datasets(data):
    """
    create_graph_datasets return the graph version of the test, validation and 
    testing datasets of the data given by filepath

    :param filepath: the path to the data
    :return: train, valid and test graph datasets  
    """

    def extract_node_features(index):
        features = torch.reshape(data.x[index, :], (num_features, data.N-1))
        features = features.t()
        features = torch.cat((torch.zeros(1, num_features), features), 0)
        return features

    edge_index = edge_index_calculation(data.A)
    num_features = 2

    dataset_train = []
    dataset_valid = []
    dataset_test = []

    for i in range(data.train_idx[0], data.train_idx[1]):
        features = extract_node_features(i)
        graph_data = Data(x=features.float(), edge_index=edge_index, idx=i,
                          y=data.trainY[i, :])
        # graph_data.validate(raise_on_error=True)
        dataset_train.append(graph_data)

    for i in range(data.valid_idx[0], data.valid_idx[1]):
        features = extract_node_features(i)
        graph_data = Data(x=features.float(), edge_index=edge_index, idx=i,
                          y=data.validY[i-data.valid_idx[0], :])
        dataset_valid.append(graph_data)

    for i in range(data.test_idx[0], data.num):
        features = extract_node_features(i)
        graph_data = Data(x=features.float(), edge_index=edge_index, idx=i,
                          y=data.testY[i-data.test_idx[0], :])
        dataset_test.append(graph_data)

    return dataset_train, dataset_valid, dataset_test


def edge_index_calculation(A):
    """
    edge_index_calculation calculates the Graph connectivity in COO format with 
    shape [2, 2*num_edges]

    :param A: the negative incidence matrix in sparse format
    :return: the edge index tensor with shape [2,2*num_edges]
    """

    A = A.to_dense()

    A_indexes_pos = torch.stack(torch.where(A == 1))
    A_indexes_neg = torch.stack(torch.where(A == -1))

    ingoing_vertices = A_indexes_pos[0, :]
    ingoing_vertices_ordered = ingoing_vertices[A_indexes_pos[1, :]]

    outgoing_vertices = A_indexes_neg[0, :]
    outgoing_vertices_ordered = outgoing_vertices[A_indexes_neg[1, :]]

    # Graph connectivity in COO format with shape [2, num_edges] (directed graph)
    edge_index_directed = torch.stack((ingoing_vertices_ordered,
                                       outgoing_vertices_ordered))

    # Graph connectivity in COO format with shape [2, 2*num_edges] (undirected graph)
    edge_index_undirected = torch.cat((edge_index_directed,
                                       torch.flip(edge_index_directed, (0,))), 1)

    return edge_index_undirected


def train(model, optimizer, criterion, loader, args, utils):
    model.train()

    epoch_loss = 0
    i=0
    for data in loader:
        z = model(data.x, data.edge_index, training=True)
        x_nn = xgraph_xflatten(data.x, args['batchSize'])
        z_hat, zc_hat = utils.complete(x_nn, z)

        Znew_train, ZCnew_train = grad_steps(x_nn, z_hat, zc_hat, args, utils,
                                             data.idx, plotFlag=False)
        train_loss, soft_weight = total_loss(x_nn, Znew_train, ZCnew_train,
                                             criterion, utils, args, data.idx,
                                             train=True)

        # all_soft_weights.append(soft_weight.tolist())
        # all_soft_weights.append(soft_weight if isinstance(soft_weight, int) else soft_weight.tolist())

        train_loss.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss = epoch_loss + train_loss.sum()
        i += 1
    return epoch_loss/i


def test_or_validate(model, criterion, loader, args, utils):
    model.eval()

    test_loss_total = 0
    i=0
    for data in loader:
        z = model(data.x, data.edge_index, training=False)
        x_nn = xgraph_xflatten(data.x, 400)
        z_hat, zc_hat = utils.complete(x_nn, z)

        Znew_train, ZCnew_train = grad_steps(x_nn, z_hat, zc_hat, args, utils,
                                             data.idx, plotFlag=False)
        test_loss, soft_weight = total_loss(x_nn, Znew_train, ZCnew_train,
                                             criterion, utils, args, data.idx,
                                             train=False)
        test_loss_total += test_loss.sum()
        i += 1
    return test_loss_total/2*i


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
                ineq_cost = torch.norm(utils.ineq_dist(
                    z_new.detach(), zc_new.detach(), idx), dim=1)
                ineq_costs = np.hstack(
                    (ineq_costs, ineq_cost[0].detach().numpy()))
                ineq_cost_c = torch.norm(
                    utils.ineq_dist(z_new_c, zc_new_c, idx), dim=1)
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
    def __init__(self, GNN, readout):
        super().__init__()
        self.GNN = GNN
        self.readout = readout

    def forward(self, x, graph, training):
        x = self.GNN(x, graph)
        if training:
            x_nn = xgraph_xflatten(x, 200, first_node=True)
        else:
            x_nn = xgraph_xflatten(x, 400, first_node=True)
        x = self.readout(x_nn)
        return x


if __name__ == '__main__':
    main()
