import argparse
import time
from GNNs import Train_gcn_model, Evaluate_gcn_model
from data_process import Get_data
from GNN_Explainer import Explain_model



def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default = 'explain', help = "Setting the mode type. (train / explain)")
    parser.add_argument("--device", default = 'cpu', help="Setting the task type. (train / explain)")

    # gnn模型“训练”阶段的参数设置
    parser.add_argument("--dataset", default = 'Tree_Cycle', help = "Set the datasets. (BA_shapes / )")
    parser.add_argument("--epoch", default = 5000, help = "Epoch, in training stage. (A number such as 100)")
    parser.add_argument("--batch_size", default = 32, help = "Batch size, in training stage. (A number such as 32)")
    parser.add_argument("--lr", default = 0.01, help = "Learn rate. (A number such as 0.001)")


    # gnn模型“解释”阶段的参数设置
    parser.add_argument("--epoch_E", default = 5000, help="Epoch, in explanation stage. (A number such as 100)")
    parser.add_argument("--batch_size_E", default=256, help="batch_size, in explanation stage. (A number such as 32)")
    parser.add_argument("--lr_E", default=0.001, help="Learn rate, in explanation stage. (A number such as 0.001)")
    parser.add_argument("--initNodeNum", default=10, type=int, help="The number of nodes of initialzed graph . (A number such as 16)")
    parser.add_argument("--explain_class", default=1, help="Categories that require explanation. (A number such as 0)")
    parser.add_argument("--final_node_number", default=7, help="The final node number of the explanation")


    return parser.parse_args()




if __name__ == '__main__':

    args = arg_parse()
    re_run = True

    if args.mode == 'train':
        Train_gcn_model(args)
    elif args.mode == 'explain':
        start_time = time.time()
        while (re_run):
            re_run = Explain_model(args)
        end_time = time.time()
        print('time consumption:',end_time-start_time)



