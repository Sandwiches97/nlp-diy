import torch
from torch import nn
from torch.nn import functional as F

class CenterdLayer(nn.Module):
    def __init__(self):
        super(CenterdLayer, self).__init__()

    def forward(self, X):
        return X - X.mean()

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bais = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bais.data
        return F.relu(linear)


def main():
    pass
    print(torch.randn(10, 3))
    print(torch.randn(3, ).shape)

    linear = MyLinear(5, 3)
    linear(torch.rand(2, 5))








if __name__ == "__main__":
    main()