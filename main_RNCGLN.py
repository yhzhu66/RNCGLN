import numpy as np
import random as random
import torch
import copy
import argparse
from models.Semi_RNCGLN import RNCGLN

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Cora', help='dataset used.')
parser.add_argument('--r1', type=float, default=1, help='hyparameter in loss function.')
parser.add_argument('--tau', type=float, default=1, help='hyparameter in contrastive loss.')
parser.add_argument('--order', type=int, default=4, help='number of multi-hop graph.')
parser.add_argument('--nb_epochs', type=int, default=5000, help='maximal epochs.')
parser.add_argument('--patience', type=int, default=40, help='early stop.')
parser.add_argument('--nheads', type=int, default=8, help='number of heads in self-attention.')
parser.add_argument('--Trans_layer_num', type=int, default=2, help='layers number for self-attention.')
parser.add_argument('--lr', type=float, default=0.0005, help='learning ratio.')
parser.add_argument('--MLPdim', type=int, default=128, help='hidden dimension.')
parser.add_argument('--trans_dim', type=int, default=128, help='hidden dimension for transformer.')
parser.add_argument('--dropout_att', type=float, default=0.4, help='dropout in self-attention layers.')
parser.add_argument('--random_aug_feature', type=float, default=0.4, help='dropout in hidden layers.')
parser.add_argument('--wd', type=float, default=5e-4, help='weight delay.')
parser.add_argument('--act', type=str, default='leakyrelu', help='hidden action.')
# for noise generation
parser.add_argument('--lab_noi_rat', type=float, default=0.3, help='noise ratio for label.')
parser.add_argument('--noise', type=str, default='uniform', help='the type for label noise generation.')
parser.add_argument('--gra_noi_rat', type=float, default=0.1, help='noise ratio for graph.')
parser.add_argument('--pres_label', type=bool, default=False, help='indicate whether original lables are preserved in the psuedo labels')
parser.add_argument('--warmup_num', type=int, default=200, help='epoch for warm-up.')
parser.add_argument('--IsLabNoise', type=bool, default=True, help='indicate the use of label-self-improvement')
parser.add_argument('--SamSe', type=bool, default=False, help='indicate the use of node selection.')
parser.add_argument('--P_sel', type=float, default=0.9, help='ratio to preserve 1-0.9 pseudo labels.')
parser.add_argument('--P_sel_onehot', type=float, default=0.95, help='ratio to preserve 1-0.9 one-hot label.')
parser.add_argument('--IsGraNoise', type=bool, default=True, help='is there label noise')
parser.add_argument('--P_gra_sel', type=float, default=0.9, help='ratio for preserve 1-0.99 edges for pseudo graph.')

args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    if torch.cuda.is_available():
        args.device = torch.device('cuda:4')
    else:
        args.device = torch.device('cpu')

    ACC_seed = []
    Time_seed = []
    for seed in range(2020, 2024):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        embedder = RNCGLN(copy.deepcopy(args))
        test_acc, training_time, stop_epoch = embedder.training()
        ACC_seed.append(test_acc)
        Time_seed.append(training_time)
        torch.cuda.empty_cache()
    ACC_seed = np.array(ACC_seed)*100

    print("-->ACC %.4f  -->STD is: %.4f" %(np.mean(ACC_seed), np.std(ACC_seed)))
