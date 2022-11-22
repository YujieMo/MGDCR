import os
import torch.nn as nn
from tqdm import tqdm
from evaluate import evaluate
from embedder import embedder
import numpy as np
import random as random
import torch
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
from models import LogReg



class MGDCR(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.criteria = nn.BCEWithLogitsLoss()
        self.cfg = args.cfg
        self.sigm = nn.Sigmoid()
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # # ===================================================#
        xent = nn.CrossEntropyLoss()
        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        print("Started training...")
        model = trainer(self.args)
        model = model.to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        model.train()

        for epoch in tqdm(range(self.args.nb_epochs+1)):
            optimiser.zero_grad()
            loss, semi = model(features, adj_list)
            if self.args.isSemi:
                semi_loss = xent(semi[self.idx_train], torch.argmax(self.labels[self.idx_train], dim=1))
                loss += semi_loss

            loss.backward()
            optimiser.step()
        # torch.save(model.state_dict(), 'saved_model/best_{}_{}.pkl'.format(self.args.dataset,self.args.custom_key))
        if self.args.use_pretrain:
            model.load_state_dict(torch.load('saved_model/best_{}_{}.pkl'.format(self.args.dataset,self.args.custom_key)))
        print('loss', loss)
        print("Evaluating...")
        model.eval()
        hf = model.embed(features, adj_list)
        macro_f1s, micro_f1s, k1, st = evaluate(hf, self.idx_train, self.idx_val, self.idx_test, self.labels,task=self.args.custom_key,epoch = self.args.test_epo,lr = self.args.test_lr,iterater=self.args.iterater) #,seed=seed
        return macro_f1s, micro_f1s, k1, st




def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class trainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cfg = args.cfg
        self.MLP1 = make_mlplayers(args.ft_size, args.cfg)
        self.MLP2 = make_mlplayers(args.ft_size, args.cfg)
        self.MLP3 = make_mlplayers(args.ft_size, args.cfg)
        length = args.length
        self.w_list = nn.ModuleList([nn.Linear(cfg[-1], cfg[-1], bias=True) for _ in range(length)])
        self.y_list = nn.ModuleList([nn.Linear(cfg[-1], 1) for _ in range(length)])
        self.W = nn.Parameter(torch.zeros(size=(length * cfg[-1], cfg[-1])))
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)
        if args.isSemi:
            self.logistic = LogReg(cfg[-1], args.nb_classes).to(args.device)
        self.encoder = nn.ModuleList()
        self.encoder.append(self.MLP1)
        self.encoder.append(self.MLP2)
        self.encoder.append(self.MLP3)



    def combine_att(self, h_list):
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h

    def forward(self, x, adj_list=None):
        x = F.dropout(x, self.args.dropout, training=self.training)
        h_p_list = []
        h_a_list = []
        for i in range(self.args.length):
            h_a = self.encoder[i](x)
            if self.args.sparse:
                h_p = torch.spmm(adj_list[i], h_a)
            else:
                h_p = torch.mm(adj_list[i], h_a)
            h_a_list.append(h_a)
            h_p_list.append(h_p)

        if self.args.isSemi:
            h_fusion = self.combine_att(h_p_list)
            semi = self.logistic(h_fusion).squeeze(0)
        else:
            semi = 0
        loss_inter = 0
        loss_intra = 0
        for i in range(self.args.length):
            intra_c = (h_p_list[i]).T @ (h_a_list[i])
            on_diag_intra = torch.diagonal(intra_c).add_(-1).pow_(2).sum()
            off_diag_intra = off_diagonal(intra_c).pow_(2).sum()
            loss_intra += (on_diag_intra + self.args.lambdintra[i] * off_diag_intra) * self.args.w_intra[i]
            if i == 1 and self.args.length == 2:
                break
            inter_c = (h_p_list[i]).T @ (h_p_list[(i + 1) % self.args.length])
            on_diag_inter = torch.diagonal(inter_c).add_(-1).pow_(2).sum()
            off_diag_inter = off_diagonal(inter_c).pow_(2).sum()
            loss_inter += (on_diag_inter + self.args.lambdinter[i] * off_diag_inter) * self.args.w_inter[i]

        loss = loss_intra + loss_inter
        return loss, semi

    def embed(self, x, adj_list=None):
        h_p_list = []
        for i in range(self.args.length):
            h_a = self.encoder[i](x)
            if self.args.sparse:
                h_p = torch.spmm(adj_list[i], h_a)
            else:
                h_p = torch.mm(adj_list[i], h_a)
            h_p_list.append(h_p)
        h_fusion = self.combine_att(h_p_list)

        return  h_fusion.detach()


def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)#, result

