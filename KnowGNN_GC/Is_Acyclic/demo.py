import argparse
import torch
from GNNs import Train_gcn_model
import time
from GNN_Explainer import Explain_model
from dateset import IsAcyclicDataset



if __name__ == '__main__':

    data = IsAcyclicDataset('data', name='Is_Acyclic')
    absence_edge_ratio_sum = 0.0
    for i in data:
        single_absence_edge_ratio = len(i.edge_index[0])/(len(i.x)*(len(i.x)-1))
        absence_edge_ratio_sum+=single_absence_edge_ratio
    absence_edge_ratio = absence_edge_ratio_sum/len(data)
    print(absence_edge_ratio)

