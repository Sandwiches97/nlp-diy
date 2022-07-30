import torch
from torch import nn
from d2l_en.pytorch.d2l import torch as d2l

def corr2d(X, K):
    """ calculate the 2-dim cross-correlation
    尺寸变换：
        - 输入：x (m, n), kernel (a, b)
        - 输出：(m - (a-1), n - (b-1))，即缩减了 (a-1, b-1)
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i+h, j: j+w] * K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

def main():


    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)

    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat-Y)**2
        conv2d.zero_grad()
        l.sum().backward()
        conv2d.weight.data[:] -= lr*conv2d.weight.grad
        if (i+1)%2==0:
            print(f"epoch {i+1}, loss {l.sum(): .3f}")

    print(conv2d.weight.data.reshape((1, 2)))

if __name__=="__main__":
    main()