from torch_geometric.datasets import TUDataset
import torch
import networkx as nx



if __name__ == '__main__':

    data = TUDataset('data/TUDataset', name = 'MUTAG')
    n=0
    for i in range(len(data)):
        if len(data[i].x)>=n:
            n = len(data[i].x)
    print(n)

