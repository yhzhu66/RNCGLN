import torch
from utils import process
from termcolor import cprint

class embedder_single:
    def __init__(self, args):
        cprint("## Loading Dataset ##", "yellow")
        adj_list, features, labels, idx_train, idx_val, idx_test = process.load_single_graph(args)
        features = process.preprocess_features(features)

        args.nb_nodes = adj_list[0].shape[0]
        args.nb_classes = int(labels.max() - labels.min()) + 1
        args.ft_size = features.shape[1]

        self.adj_list = adj_list
        self.dgl_graph = process.torch2dgl(adj_list[0])
        self.features = torch.FloatTensor(features)
        self.labels = labels.to(args.device)
        self.idx_train = idx_train.to(args.device)
        self.idx_val = idx_val.to(args.device)
        self.idx_test = idx_test.to(args.device)

        self.args = args


