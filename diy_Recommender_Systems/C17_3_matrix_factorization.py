import numpy as np
import torch
from torch import nn
from utilts import *
from d2l_en.pytorch.d2l import torch as d2l

class MF(nn.Module):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_q = self.item_bias(item_id)
        outputs = (P_u*Q_i).sum(axis=1) + torch.squeeze(b_u) + torch.squeeze(b_q)
        return outputs.flatten()

def evaluator(net, test_iter, devices):
    loss = nn.MSELoss()
    rmse = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        r_hat = [net(u.to(devices), i.to(devices)) for u, i in zip(users, items)]
        rmseLoss = torch.sqrt(loss(r_hat, ratings))
        rmse.append(rmseLoss)
    return float(np.mean(rmse))

def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=try_gpu(), evaluator=None, **kwargs):
    timer = Timer()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    net.train()
    for epoch in range(num_epochs):
        metric, l = Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(v.to(devices))
            train_feat = input_data[:-1] if len(values)>1 else input_data
            train_label = input_data[-1]




def main():
    pass


if __name__=="__main__":
    main()