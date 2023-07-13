import os.path as osp
from load_datasets import get_dataset
import torch





def Get_data(args):
    print('loading data......')
    dirs = osp.join('data/TUDataset', args.dataset, 'processed', 'data.pt')
    if not osp.exists(dirs):
        get_dataset(dataset_dir='data/TUDataset', dataset_name=args.dataset, task=None)
    data_dir = osp.join(dirs)
    dataset = torch.load(data_dir)
    data = dataset[0]
    if data.edge_attr == None:
        edge_att = torch.ones(data.num_edges, dtype=torch.float)
        data.edge_attr = edge_att


    if args.dataset == 'BA_shapes':
        for i,yy in enumerate(data.y):
            if yy>0:
                data.y[i]=1


    # print(type(data))
    # print(len(data))
    # print(data.test_mask.sum().item())
    return data



if __name__ == '__main__':
    print()
