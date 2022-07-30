import torch
from torch import nn
from torch.nn import functional as F

class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例，我们把它保存在“Modlue” 类的成员;
            # 变量 _modules 中，module的类型是OrderedDict
            self.__modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict 保证了按照成员添加顺序，遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        # 不计算梯度的随机权重参数，因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数，以及relu 和 mm 函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 使用全连接层，这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

def main():
    pass
    net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

    X = torch.rand(2, 20)
    print(net(X))

    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    chimera(X)

if __name__ == "__main__":
    main()