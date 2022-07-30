import numpy as np
import torch
from torch.utils import data
from d2l_en.pytorch.d2l import torch as d2l
from typing import Tuple
from torch import nn

def load_array(data_arrays: Tuple[list, list], batch_size, is_train=True):
    """ construct a pytorch iterator """
    dataset = data.TensorDataset(*data_arrays) # 类似 zip函数
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2,1))

    # 初始化模型参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epoch = 3
    for epoch in range(num_epoch):
        for X, y in data_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()          # 更新参数
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

if __name__ == "__main__":
    main()