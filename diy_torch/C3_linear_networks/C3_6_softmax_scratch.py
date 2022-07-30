from IPython import display
import torch
from d2l_en.pytorch.d2l import  torch as d2l
from diy_torch.C3_linear_networks.C3_2_LR_scratch import sgd



def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 此处应用了广播机制

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    """For accumulating sums over `n` variables., 为变量计数用的"""
    def __init__(self, n):
        self.data = [0.0]*n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel()) # tensor().numel() return int, 表示张量中元素总个数
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    # 将模型设置为 训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练精确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()                     # 向量的反向传播，不过提前算了均值
            updater.step()
        else:
            # 使用 自定义 的优化器 和 loss function
            l.sum().backward()                      # 向量的反向传播
            updater(X.shape[0])                   # 对整个batchsize更新参数
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale="linear", yscale="linear", fmts=('-', "m--", "g-.", "r:"),
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axis = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows*ncols == 1:
            self.axes = [self.axis, ]
        # 使用lambda 函数捕获参数
        self.config_axes = lambda : d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """ 训练模型 """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics+(test_acc, ))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def main():


    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    def net(X):
        return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

    def cross_entropy(y_hat, y):
        """选则 one-hot 编码 y 相应的标签的位置 idx_list，返回其 -torch.log(list, idx_list]

        :param y_hat: (batch size, num_class)
        :param y: (batch size, num_class)
        :return: (batch size)
        """
        return -torch.log(y_hat[range(len(y_hat)), y])

    def show_soft():
        X = torch.normal(0, 1, (2, 5))
        X_prob = softmax(X)
        print(X_prob, "\n", X_prob.sum(1))
    show_soft()
    def show_loss():
        y = torch.tensor([0, 2])
        y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
        print(y_hat[[0, 1], y])
    show_loss()
    lr = 0.1
    def updater(batch_size):
        return sgd([W, b], lr, batch_size)

    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

if __name__ == "__main__":
    main()