import numpy as np
import torch
from torch import nn
from utilts import *
from d2l_en.pytorch.d2l import torch as d2l
from diy_Recommender_Systems.C17_2_ml_dataset import split_and_load_ml100k

class MF(nn.Module):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors) # nn.Embedding 是无偏置的仿射变换
        self.Q = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        user_id = user_id.to(dtype=torch.long)
        item_id = item_id.to(dtype=torch.long)
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
        # r_hat = [net(u.to(devices), i.to(devices)) for u, i in zip(users, items)]
        r_hat = net(users.to(devices), items.to(devices))
        rmseLoss = torch.sqrt(loss(r_hat, ratings.to(devices)))
        rmse.append(rmseLoss)
    return float(np.mean([float(it) for it in rmse]))

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
            # input_data = List[user ids, item ids, ratings]
            train_feat = input_data[:-1] if len(values)>1 else input_data
            train_label = input_data[-1]

            l = 0.
            preds = net(train_feat[0], train_feat[1])
            ls = loss(preds, train_label.to(dtype=torch.float))
            ls.backward()
            l += ls.item()
            # preds = [net(t) for t in zip(*train_feat)]
            # ls = [loss(p, s.to(dtype=torch.float)) for p, s in zip(preds, train_label)]
            # for l in ls:
            #     l.backward()
            # l += sum([l.data for l in ls]).mean()
            trainer.step()
            metric.add(l, values[0].shape[0], values[0].numel())
            timer.stop()
        if len(kwargs) > 0:
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'], devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l/(i+1)
        animator.add(epoch+1, (float(train_l), test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')

def main():
    devices = d2l.try_all_gpus()
    num_users, num_items, train_iter, test_iter = split_and_load_ml100k(
        test_ratio=0.1, batch_size=512
    )
    net = MF(30, num_users, num_items).to(devices[0])
    def init_weights(m):
        if type(m)==nn.Embedding:
            nn.init.normal_(m.weight, mean=0., std=0.01)
    net.apply(init_weights)
    lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, "adam"
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    train_recsys_rating(net, train_iter, test_iter, loss, optimizer, num_epochs, devices[0], evaluator)

    # 测试 user id=20，item id=30 的评分
    scores = net(torch.Tensor([20]).to(devices[0]),
                 torch.Tensor([30]).to(devices[0]))
    print(scores)
if __name__=="__main__":
    main()