import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes, degree
from torch.distributions.categorical import Categorical
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import numpy as np
import random
from data_process import Get_data
import torch.nn.functional as F



def Get_tensor_classes_num(y_tensor):
    #获取类别数目

    return len(set(y_tensor.numpy().tolist()))



def load_checkpoint(model, checkpoint_PATH):
    #加载模型

    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    # print('loading checkpoint......')
    # optimizer.load_state_dict(model_CKPT['optimizer'])
    return model



def Initial_graph_generate(args):
    #初始化一个无向完全图

    data = Get_data(args)
    absence_edge_ratio = 1 - len(data.edge_index[0]) / (len(data.x)*degree(data.edge_index[0]).max())
    edge_category_probability_ratio = torch.tensor([absence_edge_ratio, 1-absence_edge_ratio], dtype=torch.float)
    node_category_probability_ratio = torch.tensor([1-data.y.sum().item()/len(data.x), data.y.sum().item()/len(data.x)], dtype=torch.float)
    node_category_distribution = Categorical(node_category_probability_ratio)
    edge_category_distribution = Categorical(edge_category_probability_ratio)

    edge_categories = []
    node_categories = []

    for i in range(args.initNodeNum):
        node_categories.append(node_category_distribution.sample().item())
    for i in range(int(args.initNodeNum * (args.initNodeNum - 1) / 2)):
        indicate_edge = edge_category_distribution.sample().item()
        edge_categories.append(indicate_edge)
        edge_categories.append(indicate_edge)

    # edge_index
    edge_index_complete = []
    for i in range(args.initNodeNum):
        for j in range(args.initNodeNum):
            if j > i:
                edge_index_complete.append([i, j])
                edge_index_complete.append([j, i])
    edge_index = torch.tensor(edge_index_complete, dtype=torch.long).T

    edge_categories_tensor = torch.tensor(edge_categories, dtype=torch.float)
    edge_presence_indicator = edge_categories_tensor.bool()  # 用于指示不存在的边

    edge_index = torch.stack((torch.masked_select(edge_index[0], edge_presence_indicator),
                              torch.masked_select(edge_index[1], edge_presence_indicator)))

    # edge_attr
    edge_attr = torch.masked_select(edge_categories_tensor, edge_presence_indicator).float()  # 去除不存在的边

    node_categories_tensor = torch.tensor(node_categories, dtype=torch.long)
    x = F.one_hot(node_categories_tensor, num_classes=2).squeeze(0).float()
    seed_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if seed_graph.has_isolated_nodes():
        _, _, node_mask = remove_isolated_nodes(seed_graph.edge_index, num_nodes=len(seed_graph.x))
        x = x[node_mask]
        edge_index = Fix_nodes_index(edge_index)
        seed_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(seed_graph)
    return seed_graph

def Fix_nodes_index(edge_index):
    b = edge_index.view(-1)
    c = []
    d = {}
    e = []
    for i in b:
        if i not in c:
            c.append(i.item())
    c.sort()

    for v,k in enumerate(c):
        d[k] = v

    for i in edge_index:
        for j in i :
            e.append(d[j.item()])
    t = torch.tensor(e).view(2,-1)
    return t


def Get_dataset_class_num(dataset_name):
    if dataset_name == 'BA_shapes':
        return 2
    elif dataset_name == 'Tree_Cycle':
        return 2


def Draw_graph(Data,j):
    edge_attr = []
    edge_max = 0.0
    for i in Data.edge_attr:
        edge_attr.append(i.item())
        if i.item() > edge_max:
            edge_max = i.item()
        G = to_networkx(Data, to_undirected=True, remove_self_loops=True)
    # print(edge_attr)
    pos = nx.spring_layout(G)
    for n in G.nodes:
        if n == 0:
            color = 'red'
        else:
            color = 'blue'
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=color)
    i=0
    for (u, v, d) in G.edges(data=True):

        # print(u, v, edge_attr[i])
        # G.add_edge(u, v, weight=edge_attr[i])
        nx.draw_networkx_edges(G, pos, edgelist=[(u,v)])
        i += 1
    # nx.draw(G)
    image_save_path = 'img/graph'+str(j+1)+'.png'
    plt.savefig(image_save_path)
    plt.close('all')
    # plt.show()

def make_one_hot(data1,args):
    l = Get_dataset_class_num(args.dataset)

    return (np.arange(l)==data1[:,None]).astype(np.integer)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def Random_select_0(gate):
    index_0 = []
    for i in range(len(gate)):
        if gate[i]<1:
            index_0.append(i)
    ran_max = random.randint(0,len(index_0)-1)
    ga=(gate >= 0).float()
    ga[index_0[ran_max]]=0
    return ga




def Fix_gate(gate, graph_index):
    if len(gate)-gate.sum().item()>1:
        gate = Random_select_0(gate)
        # for j in range(len(gate)-gate.sum().long().item()-1):
        #     gate[gate.argmin()]=1.0

    if len(graph_index) == 2:
        graph_index = torch.t(graph_index)
    min_index = gate.argmin()
    u = graph_index[min_index.item()][0].item()
    v = graph_index[min_index.item()][1].item()
    # inverse_uv = torch.Tensor([v,u]).to(device)
    inverse_uv = torch.Tensor([v, u])

    for i, edge in enumerate(graph_index.float()):
        if torch.equal(edge, inverse_uv):
            gate[i]=0
    return gate










