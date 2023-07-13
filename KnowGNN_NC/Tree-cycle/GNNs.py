import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from nnUtils.RGCNLayer import RGCN
from data_process import Get_data
import numpy as np
import os
import os.path as osp
from utils import Get_tensor_classes_num, load_checkpoint, Get_dataset_class_num
from torch.nn import ReLU, Linear, LayerNorm



class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = 100
        self.output_dim = output_dim

        self.injected_message_scale = None
        self.injected_message_replacement = None

        self.transform = torch.nn.Sequential(
            Linear(self.input_dim, self.hid_dim),
            LayerNorm(self.hid_dim),
            torch.nn.ReLU(),
            Linear(self.hid_dim, self.hid_dim),
            LayerNorm(self.hid_dim),
            torch.nn.ReLU(),
        )

        self.rgcn = RGCN(self.hid_dim, self.hid_dim, n_relations=1, n_layers=1, inverse_edges=False)
        self.conv1 = GCNConv(self.hid_dim, self.hid_dim)
        self.conv2 = GCNConv(self.hid_dim, self.output_dim)

    def forward(self, data_x, data_edge_index, data_edge_attr):
        # x, edge_index = data.x, data.edge_index
        x = self.transform(data_x)
        x = self.rgcn(x, data_edge_index, data_edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv1(x, data_edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, data_edge_index)

        return F.log_softmax(x, dim=1)

    def inject_message_scale(self, message_scale):
        self.rgcn.inject_message_scale(message_scale)

    def inject_message_replacement(self, message_replacement):
        self.rgcn.inject_message_scale(message_replacement)  # Have to store it in a list to prevent the pytorch module from thinking it is a parameter

    def get_vertex_embedding_dims(self):
        return self.hid_dim

    def get_message_dims(self):
        return self.hid_dim

    def get_latest_source_embeddings(self):
        return self.rgcn.get_latest_source_embeddings()
        # return [layer.get_latest_source_embeddings() for layer in self.gnn_layers]

    def get_latest_target_embeddings(self):
        return self.rgcn.get_latest_target_embeddings()
        # return [layer.get_latest_target_embeddings() for layer in self.gnn_layers]

    def get_latest_messages(self):
        return self.rgcn.get_latest_messages()
        # return [layer.get_latest_messages() for layer in self.gnn_layers]

    def count_latest_messages(self):
        return sum([layer_messages.numel() / layer_messages.shape[-1] for layer_messages in self.get_latest_messages()])






def Train_gcn_model(args):
    #训练GCN

    dataset = Get_data(args)
    model_save_dir = osp.join('checkpoint', args.dataset)
    if not osp.exists(model_save_dir):
        os.mkdir(model_save_dir)

    input_dim = dataset.num_node_features
    # out_dim = Get_tensor_classes_num(dataset['y'])
    out_dim = Get_dataset_class_num(args.dataset)
    model = GCN(input_dim, out_dim).to(args.device)

    # edge_att = torch.ones(dataset.num_edges, dtype=torch.float)
    # dataset.edges_attr = edge_att
    data = dataset.to(args.device)
    print(data)
    data_x = data.x
    data_edge_index = data.edge_index
    data_edge_attr = data.edge_attr

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    loss_list = []
    best_loss = 0
    best_acc = 0
    # model.train()
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()

        out = model(data_x, data_edge_index, data_edge_attr)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        logits, accs = model(data_x, data_edge_index, data_edge_attr), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
            loss_list.append(loss)

        if epoch % 10 == 9:
            log = 'Epoch: {:03d}, Train: {:.5f}, Train loss: {:.5f}, Val: {:.5f}, Test: {:.5f}'
            print(log.format(epoch + 1, accs[0], loss, accs[1], accs[2]))

        if accs[2]>best_acc:
            best_acc = accs[2]
            best_loss = loss
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_test_acc': best_acc,
                'loss': best_loss
            }
            model_name = args.dataset +'_gcn_model.pth'
            model_save_path = osp.join(model_save_dir,model_name)
            torch.save(state, model_save_path)
            print('New model has saved. Model test acc is up to: {:.5f}'.format(best_acc))
            print()

    print('\nModel trained completed.\nbest test acc: {:.5f}    loss: {:.5f}'.format(best_acc, best_loss))





def Evaluate_gcn_model(args, datas):
    #评估、测试模型

    dataset = datas
    node_features = dataset.num_node_features
    class_num = Get_dataset_class_num(args.dataset)
    data = dataset.to(args.device)

    # print(dataset.x[dataset.test_mask])

    model = GCN(node_features,class_num).to(args.device)
    model_name = args.dataset + '_gcn_model.pth'
    model_save_path = osp.join('checkpoint', args.dataset, model_name)
    model = load_checkpoint(model, model_save_path)
    model.eval()

    if args.mode == 'evaluate':
        p = model(data.x, data.edge_index, data.edge_attr)
        pred = p.argmax(dim=1)
        print(p)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        print('Accuracy: {:.4f}'.format(acc))
    elif args.mode == 'explain':
        pred = model(data.x, data.edge_index, data.edge_attr)
        pred_exp = torch.exp(pred, out=None)

        pred_exp_3 = pred_exp.unsqueeze(1)
        pred_exp_max = F.max_pool1d(pred_exp_3,kernel_size=class_num)
        pred_exp_max_sum = pred_exp_max.squeeze(1).sum()
        # print(pred_exp_max_sum)

        return pred_exp_max_sum, pred_exp.argmax(dim=-1)




