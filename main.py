"trying to implement a simple GNN"

"extern"
import torch 
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils import OPTreconfigure

"local"
from nets import GCN



def main():    
    print("Hello")

    filepath = 'node4_dataset'
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print('Network file does not exist')

    # Once we use bigger examples, we should use the Dataset Class of GPY 
    dataset_train, dataset_valid, dataset_test = create_graph_datasets(data)

    train_loader = DataLoader(dataset_train, batch_size=200, shuffle=True)
    valid_loader = DataLoader(dataset_valid, len(dataset_valid), shuffle=True)
    test_loader = DataLoader(dataset_test, len(dataset_test), shuffle=True)

    model = GCN(input_features=2, hidden_channels=12, output_classes=3, layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # TODO look how Rabab implemented her loss
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, 100):
        loss = train(model, optimizer, criterion, train_loader)
        eval_loss = test_or_validate(model, valid_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')



def create_graph_datasets(data):
    """
    create_graph_datasets return the graph version of the test, validation and 
    testing datasets of the data given by filepath

    :param filepath: the path to the data
    :return: train, valid and test graph datasets  
    """
    
    def extract_node_features(index):
        features = torch.reshape(data.x[index,:], (num_features, data.N-1))
        features = features.t()
        features = torch.cat((torch.zeros(1,num_features), features), 0)
        return features

    edge_index = edge_index_calculation(data.A)
    num_features = 2

    dataset_train = []
    dataset_valid = []
    dataset_test = []

    for i in range(data.train_idx[0], data.train_idx[1]):
        features = extract_node_features(i)
        graph_data = Data(x=features.float(), edge_index=edge_index, 
                            y=data.trainY[i, :])
        graph_data.validate(raise_on_error=True)
        dataset_train.append(graph_data)

    for i in range(data.valid_idx[0], data.valid_idx[1]):
        features = extract_node_features(i)
        graph_data = Data(x=features.float(), edge_index=edge_index, 
                            y=data.validY[i-data.valid_idx[0], :])
        dataset_valid.append(graph_data)

    for i in range(data.test_idx[0], data.num):
        features = extract_node_features(i)
        graph_data = Data(x=features.float(), edge_index=edge_index,
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
    
    ingoing_vertices = A_indexes_pos[0,:]
    ingoing_vertices_ordered = ingoing_vertices[A_indexes_pos[1,:]]

    outgoing_vertices = A_indexes_neg[0,:]
    outgoing_vertices_ordered = outgoing_vertices[A_indexes_neg[1,:]]
    
    # Graph connectivity in COO format with shape [2, num_edges] (directed graph)
    edge_index_directed = torch.stack((ingoing_vertices_ordered,
                                        outgoing_vertices_ordered))

    # Graph connectivity in COO format with shape [2, 2*num_edges] (undirected graph)
    edge_index_undirected = torch.cat((edge_index_directed, 
                                        torch.flip(edge_index_directed, (0,))),1)

    return edge_index_undirected


def train(model, optimizer, criterion, loader):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test_or_validate(model, loader):
    model.eval()
    
    for data in loader:
        out = model(data.x, data.edge_index)
        ##########################TODO###########################
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        #########################################################
    return test_acc



if __name__ == '__main__':
    main()









