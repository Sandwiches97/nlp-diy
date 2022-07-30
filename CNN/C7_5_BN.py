import torch
from torch import nn
from d2l_en.pytorch.d2l import torch as d2l


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps: float, momentum: float):
    # 通过 is_grad_enabled 来判断当前模式是 训练模型 还是 预测模式
    if not torch.is_grad_enabled():
        # 如果在 predict 模式，直接使用传入的移动平均所得的均值和方差
        X_hat = (X-moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Fully connection situation, 计算 feature dim 上的 mean 与 var
            mean = X.mean(dim=0)
            var = ((X-mean)**2).mean(dim=0)
        else:
            # 2-dim convolution situation, 计算 channel dim (axis=1)上的 mean 与 var
            # 这里我们需要保持 X 的形状，以便后面做 board cast 运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X-mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X-mean)/torch.sqrt(var+eps)
        # 更新 mean 与 var by moving average
        moving_mean = momentum * moving_mean + (1.0-momentum) * mean
        moving_var = momentum * moving_var + (1.0-momentum) * moving_var
    Y = gamma * X_hat + beta # scaling and moving
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        """
        :param num_features:
            - Fully Connection Layer's output nums
            或  Convolution’s out channel nums
        :param num_dims: 2 表示 FC，4 表示 Conv
        """
        super(BatchNorm, self).__init__()
        if num_dims==2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与梯度和迭代的 scale param 与 shift param，分别初始化为0，1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量，初始化为0，1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果 X 不在内存上，将moving_mean 和 moving var 复制到X的显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存 updated moving_mean 与 moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9
        )
        return Y

def main():
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
        nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
        nn.Linear(84, 10)
    )
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


if __name__=="__main__":
    main()

