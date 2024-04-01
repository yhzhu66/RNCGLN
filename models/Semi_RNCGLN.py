import time
from models.embedder import embedder_single
import torch.nn.functional as F
import torch
import torch.nn as nn
import utils.Noise_about as Noi_ab
import copy

def row_normalize(A):
    """Row-normalize dense matrix"""
    eps = 2.2204e-16
    rowsum = A.sum(dim=-1).clamp(min=0.) + eps
    r_inv = rowsum.pow(-1)
    A = r_inv.unsqueeze(-1) * A
    return A

def Ncontrast(x_dis, adj_label, tau = 1, train_index_sort=None):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum_mid = torch.sum(x_dis, 1)
    x_dis_sum_pos_mid = torch.sum(x_dis*adj_label, 1)
    x_dis_sum = x_dis_sum_mid[train_index_sort]
    x_dis_sum_pos = x_dis_sum_pos_mid[train_index_sort]
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj
    return adj_label


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Linear(in_channels, key_channels)
        self.queries = nn.Linear(in_channels, key_channels)
        self.values = nn.Linear(in_channels, value_channels)
        self.reprojection = nn.Linear(key_channels, key_channels)

    def forward(self, input_):
        keys = self.keys(input_)
        queries = self.queries(input_)
        values = self.values(input_)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:,i * head_key_channels: (i + 1) * head_key_channels], dim=0)
            query = F.softmax(queries[:,i * head_key_channels: (i + 1) * head_key_channels], dim=1)
            value = values[:,i * head_value_channels: (i + 1) * head_value_channels]
            context = key.transpose(0, 1) @ value
            attended_value = query @context
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        return attention


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class EncoderLayer(nn.Module):
    def __init__(self, args, d_model, heads, dropout=0.1):
        super().__init__()
        self.args = args
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.effectattn = EfficientAttention(in_channels = d_model, key_channels =d_model, head_count =heads, value_channels = d_model)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm_1(x)
        x_pre = self.effectattn(x2)
        x = x + x_pre
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class RNCGLN_model(nn.Module):
    def __init__(self, arg):
        super(RNCGLN_model, self).__init__()

        self.dropout = arg.random_aug_feature
        self.Trans_layer_num = arg.Trans_layer_num
        self.layers = get_clones(EncoderLayer(arg,arg.trans_dim , arg.nheads, arg.dropout_att), self.Trans_layer_num)
        self.norm_input = Norm(arg.ft_size)
        self.MLPfirst = nn.Linear(arg.ft_size,arg.trans_dim)
        self.MLPlast = nn.Linear(arg.trans_dim,arg.nclasses)
        self.norm_layer = Norm(arg.trans_dim)



    def forward(self, x_input):
        x_input = self.norm_input(x_input)
        x = self.MLPfirst(x_input)
        x = F.dropout(x, self.dropout, training=self.training)
        x_dis = get_feature_dis(self.norm_layer(x))
        for i in range(self.Trans_layer_num):
            x= self.layers[i](x)

        CONN_INDEX = F.relu(self.MLPlast(x))

        return F.softmax(CONN_INDEX, dim=1), x_dis


class RNCGLN(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.args.nclasses = (self.labels.max() - self.labels.min() + 1).item()
        self.model = RNCGLN_model(self.args).to(self.args.device)

    def training(self):

        features = self.features.to(self.args.device)
        graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)

        print("Started training...")
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay = self.args.wd)

        label_orig = self.labels.clone()
        if self.idx_train.dtype == torch.bool:
            self.idx_train = torch.where(self.idx_train == 1)[0]
            self.idx_val = torch.where(self.idx_val == 1)[0]
            self.idx_test = torch.where(self.idx_test == 1)[0]

        # self.idx_train = torch.arange(0,500).to(self.idx_train.device)
        self.label_ori = self.labels.clone()

        train_lbls = self.label_ori[self.idx_train]
        val_lbls = self.label_ori[self.idx_val]
        test_lbls = self.label_ori[self.idx_test]

        self.idx_train_ori = self.idx_train.clone()

        if self.args.lab_noi_rat != 0:
            noise_y, P = Noi_ab.noisify_with_P(train_lbls.cpu().numpy(), self.args.nclasses, noise = self.args.lab_noi_rat, random_state = 10 ,noise_type = self.args.noise)
            self.labels[self.idx_train] = torch.from_numpy(noise_y).to(self.args.device)

        ones = torch.sparse.torch.eye(self.args.nclasses).to(self.args.device)
        self.labels_oneHot = ones.index_select(0,self.labels)

        if self.args.pres_label:
            train_unsel = torch.cat((self.idx_val, self.idx_test), dim=0)
        else:
            train_unsel = torch.cat((self.idx_train,self.idx_val, self.idx_test), dim=0)

        train_all_pos_bool = torch.ones(self.labels.size(0))
        train_all_pos_bool[train_unsel] = 0
        train_all_pos = train_all_pos_bool.to(self.args.device)
        adj_label = get_A_r(graph_org_torch,self.args.order)

        cnt_wait = 0
        best = 1e-9
        output_acc = 1e-9
        stop_epoch = 0
        start = time.time()
        totalL = []
        test_acc_list = []


        PP = 0

        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()
            embeds_tra, x_dis = self.model(features) #graph_org_torch
            loss_cla = F.cross_entropy(embeds_tra[self.idx_train], self.labels_oneHot[self.idx_train])
            max_va_pos = embeds_tra.max(1)[0]
            max_va_pos_index = max_va_pos >= PP
            # max_va_pos_index = max_va_pos >= 0
            loss_Ncontrast = Ncontrast(x_dis, adj_label, tau=self.args.tau, train_index_sort = max_va_pos_index)
            loss = loss_cla + self.args.r1 * loss_Ncontrast


            loss.backward()

            optimiser.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0:
                totalL.append(loss.item())
                self.model.eval()

                embeds, _ = self.model(features)
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)
                print("{:.4f}|".format(test_acc.item()), end="") if epoch % 50 != 0 else print("{:.4f}|".format(test_acc.item()))

                test_acc_list.append(test_acc.item())
                # early stop
                stop_epoch = epoch
                if val_acc > best:
                    best = val_acc
                    output_acc = test_acc.item()
                    cnt_wait = 0
                    better_inex = 1

                    # if better_num > self.args.warmup_num and self.args.IsLabNoise:
                    if epoch > self.args.warmup_num and self.args.IsLabNoise and better_inex:
                    # if better_num > self.args.warmup_num and self.args.IsLabNoise and training_turn_ind:
                        pre_value_max, pre_index_max = embeds.max(1)
                        self.labels_oneHot = embeds.detach().clone()
                        Y_zero = torch.zeros_like(self.labels_oneHot)
                        Y_zero.scatter_(1,pre_index_max.unsqueeze(1),1)
                        self.labels_oneHot[pre_value_max >= self.args.P_sel_onehot] = Y_zero[pre_value_max >= self.args.P_sel_onehot]
                        pre_ind_min = pre_value_max >=self.args.P_sel
                        self.idx_train = pre_ind_min.float() +train_all_pos.float() == 2

                        ## update some hype-parameters
                        if not self.args.IsGraNoise:
                            better_inex = 0
                        if self.args.SamSe:
                            PP =self.args.P_sel_onehot
                        # if self.args.IsGraNoise:
                        #     training_turn_ind = 0
                        print("\n --> A new loop after label update")

                    # if better_num > self.args.warmup_num and self.args.IsGraNoise and not training_turn_ind:
                    if epoch > self.args.warmup_num and self.args.IsGraNoise and better_inex:
                        ## update GRAPH
                        x_dis_mid = x_dis.detach().clone()

                        x_dis_mid = x_dis_mid * adj_label
                        val_, pos_ = torch.topk(x_dis_mid, int(x_dis_mid.size(0) * (1 - self.args.P_gra_sel)))
                        adj_label = torch.where(x_dis_mid > val_[:, -1].unsqueeze(1), x_dis_mid, 0)

                        if self.args.SamSe:
                            PP =self.args.P_sel_onehot

                        print("\n --> A new loop after graph update")

                else:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    break

            ################END|Eval|###############

        training_time = time.time() - start
        print("\n [Classification] ACC: {:.4f} | stop_epoch: {:}| training_time: {:.4f} \n".format(
            output_acc, stop_epoch, training_time))


        return output_acc, training_time, stop_epoch

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)