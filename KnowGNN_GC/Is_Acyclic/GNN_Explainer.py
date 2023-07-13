import torch
import networkx as nx
from torch.nn import ReLU, Linear, LayerNorm
from nnUtils.hard_concrete import HardConcrete
from nnUtils.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from nnUtils.lagrangian_optimization import LagrangianOptimization
from nnUtils.squeezer import Squeezer
from nnUtils.SA import P_t_SA
from GNNs import GCN
from torch_geometric.utils.convert import to_networkx
import os.path as osp
from torch_geometric.data import Data
from utils import Initial_graph_generate, Fix_nodes_index,Get_dataset_class_num, Draw_graph, Fix_gate
import torch.nn.functional as F
from torch_geometric.utils import remove_isolated_nodes
from torch.nn import CrossEntropyLoss

from networkx.algorithms.components import number_connected_components
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Explainer(torch.nn.Module):
    def __init__(self, G, model):
        super().__init__()
        self.graph = G
        self.gnn_model = model
        self.gnn_model.eval()
        vertex_embedding_dims = self.gnn_model.get_vertex_embedding_dims()
        message_dims = self.gnn_model.get_message_dims()


        gate_input_shape = [vertex_embedding_dims, message_dims, vertex_embedding_dims]  # [100, 100, 100]

        self.mask_learing_nn = torch.nn.Sequential(
            MultipleInputsLayernormLinear(gate_input_shape, 100),
            ReLU(),
            Linear(100, 1),
            Squeezer(),
            HardConcrete()
        )


    def forward(self):

        latest_source_embeddings = self.gnn_model.get_latest_source_embeddings()  # 获取最新的源节点嵌入，在gnn处理时已经将值传递至gnn对象中，这里直接获取即可
        latest_messages = self.gnn_model.get_latest_messages()  # 获取最新的边嵌入
        latest_target_embeddings = self.gnn_model.get_latest_target_embeddings()  # 获取最新的目标节点嵌入
        gate_input = [latest_source_embeddings[0], latest_messages[0], latest_target_embeddings[0]]  # 初始化gates生成器的输入，每条边的源节点、边信息。目标节点
        gate, penalty = self.mask_learing_nn(gate_input)

        return gate, penalty


    def reset_my_parameters(self):
        for m in self.mask_learing_nn.modules():
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()


def run_discriminator(model, data, args):
    model.eval()
    class_num = Get_dataset_class_num(args.dataset)
    pred = model(data.x, data.edge_index, data.edge_attr)
    pred_exp = torch.exp(pred, out=None)

    pred_exp_3 = pred_exp.unsqueeze(1)
    pred_exp_max = F.max_pool1d(pred_exp_3, kernel_size=class_num)
    pred_exp_max_sum = pred_exp_max.squeeze(1).sum()
    # print(pred_exp_max_sum)

    return pred_exp, pred_exp.argmax(dim=-1)



def Explain_model(args):
    # args = arg_parse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G = Initial_graph_generate(args)
    gnn_loss = CrossEntropyLoss()
    tar_class = args.explain_class


    threshold = 0.97
    P_t = threshold
    P_all = [P_t]
    node_num_p = 0
    itreation = 1
    prebability = 0.0
    allowance = 0.03
    last_loss = 100.0
    last_G = Data(x=G.x, edge_index=G.edge_index, edge_attr=G.edge_attr)
    final_node_num = args.final_node_number

    final_node_num_list = []

    T = 0.0002
    roll_back_count = 0
    rerun_index = False
    while len(G.x) >= final_node_num:

        P_J = []
        gnn_model = GCN(10,2)
        model_name = args.dataset + '_gcn_model.pth'
        model_save_path = osp.join('checkpoint', args.dataset, model_name)
        # gnn_model = load_checkpoint(gnn_model, model_save_path)
        gnn_model.load_state_dict(torch.load(model_save_path))

        explainer = Explainer(G,gnn_model)

        # gnn_loss = BCEWithLogitsLoss(reduction="none")
        # target_label = torch.full(size = (1,len(G.x)), fill_value=0)                   #arg+++++
        # target_label = F.one_hot(target_label, num_classes=2).squeeze(0).float()

        target_label = torch.full(size=(1, 1), fill_value=tar_class)  # arg+++++
        target_label = target_label.squeeze(0).long()


        explainer_optimizer = torch.optim.Adam(explainer.parameters(), lr=0.001)


        lagrangian_optimization = LagrangianOptimization(explainer_optimizer, device)

        e_count = 0
        is_roll_withsame_G = False
        for epoch in range(1000):

            if epoch>=999:
                e_count=1
                break

            gnn_model.eval()
            batch = torch.zeros((1, len(G.x)), dtype=torch.int64)[0]
            pe = gnn_model(G.x, G.edge_index, G.edge_attr, batch)
            # print(torch.exp(pe, out=None))
            explainer.train()
            gate, penalty = explainer()
            gate = Fix_gate(gate, G.edge_index)
            explainer.gnn_model.inject_message_scale(gate)
            # explainer.gnn_model.inject_message_replacement(baseline)
            gnn_model.eval()
            pred = gnn_model(G.x, G.edge_index, G.edge_attr, batch)
            # pred = torch.exp(pred, out=None)

            if pred[0][tar_class].item()>threshold:
                P_J.append(pred[0].max().item())

            loss_gnn = gnn_loss(pred, target_label)

            g = torch.relu(loss_gnn - allowance).mean()
            f = penalty
            loss = lagrangian_optimization.update(f,g)

            if epoch == 0:
                log = 'Epoch: {:03d}, Train loss: {:.5f}, prediction: {:.5f}'
                print('\n{0}-th iteration...'.format(itreation))
                itreation += 1
                print(log.format(epoch + 1, loss, pred[0].max()))
            elif epoch % 10 == 9:
                log = 'Epoch: {:03d}, Train loss: {:.5f}, prediction: {:.5f}'
                print(log.format(epoch + 1, loss, pred[0].max()))

            if loss <= last_loss:
                last_loss = loss
                last_gate = gate


            elif loss > last_loss and pred[0][tar_class] >= P_t:
                masks = gate.bool()
                G.edge_index = G.edge_index[torch.stack((masks, masks))].view(2, -1)
                G.edge_index = Fix_nodes_index(G.edge_index)
                G.edge_attr = G.edge_attr[masks]
                G_nx = to_networkx(G, to_undirected=True, remove_self_loops=True)
                _, _, node_mask = remove_isolated_nodes(G.edge_index, num_nodes=len(G.x))

                if nx.is_connected(G_nx):
                    last_G = Data(x=G.x, edge_index=G.edge_index, edge_attr=G.edge_attr)
                elif node_mask.long().sum().item() < len(G.x) and number_connected_components(G_nx)==2 and node_mask.long().argmin() != 0:
                    G.x = G.x[node_mask]
                    # target_label = target_label[node_mask]
                    last_G = Data(x=G.x, edge_index=G.edge_index, edge_attr=G.edge_attr)
                else:
                    G = Data(x=last_G.x, edge_index=last_G.edge_index, edge_attr=last_G.edge_attr)
                    is_roll_withsame_G = True

                if is_roll_withsame_G:
                    roll_back_count += 1


                if len(G.x) >= final_node_num:
                    node_num_p = len(G.x)
                    prebability = pred[0][tar_class].item()
                last_loss=100
                if len(G.x) == final_node_num:
                    final_node_num_list.append(Data(x=G.x, edge_index=G.edge_index, edge_attr=G.edge_attr))
                break
        if roll_back_count > 15:
            rerun_index = True
            roll_back_count = 0
            break

        if is_roll_withsame_G == False:
            roll_back_count = 0

        if e_count == 1:
            P_t = 0.97
            T = 0.0002
        else:
            P_A = sum(P_J) / len(P_J)
            P_t = P_t_SA(P_t, P_A, T)
            T = 0.95 * T
        # if P_A >= P_t:
        #     P_t = P_A
        P_all.append(P_t)
        explainer.reset_my_parameters()
    if rerun_index:
        return rerun_index
    else:
        print('Node number of current graph: ', node_num_p)
        print('Probability: ', prebability)
        print('Probability threshold values: ', P_all)
        for i,G_l in enumerate(final_node_num_list):
            Draw_graph(G_l,i)
        return rerun_index
