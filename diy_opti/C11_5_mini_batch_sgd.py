import numpy as np
import torch
from torch import nn
from d2l_en.pytorch.d2l import torch as d2l
from diy_torch.C3_linear_networks.C3_1_LR import Timer
from diy_torch.C3_linear_networks.C3_2_LR_scratch import squared_loss

d2l.DATA_HUB["arifoil"] = (d2l.DATA_URL + "airfoil_self_noise.dat", )

def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download("airfoil"),
                         dtype=np.float32, delimiter="\t")
    data = torch.from_numpy((data-data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1

def sgd(params, states, hyperparams: dict):
    for p in params:
        p.data.sub_(hyperparams['lr']*p.grad)
        p.grad.data.zero_()

def train_ch11(trainer_fn, states, hyperparams, data_iter, feature_dim, num_epochs=2):
    # 初始化模型
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1), requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), squared_loss
    # train model
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n%200==0:
                timer.stop()
                animator.add(n / X.shape[0] / len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]

def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs
    )

def main():
    timer = Timer()
    A = torch.zeros(256, 256)
    B = torch.randn(256, 256)
    C = torch.randn(256, 256)




    # 逐元素计算矩阵乘法 A = BC
    for i in range(256):
        for j in range(256):
            A[i, j] = torch.dot(B[i, :], C[:, j])
    print(timer.stop())

    # 逐列计算 A = BC
    timer.start()
    for j in range(256):
        A[:, j] = torch.mv(B, C[:, j]) # matrix @ vector
    print(timer.stop())

    # 一次性计算 A = BC
    timer.start()
    A = torch.mm(B, C) # matrix @ matrix
    print(timer.stop())

    # 乘法和加法作为单独的操作（在实践中融合）
    gigaflops = [2/i for i in timer.times if i>0 ]
    print(f"performance in Gigaflops: element {gigaflops[0]: .3f}, "
          f"column {gigaflops[1]}")

    gd_res = train_sgd(1, 1500, 100)

if __name__=="__main__":
    main()