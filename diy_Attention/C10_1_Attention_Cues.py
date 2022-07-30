import torch
from torch import nn
from d2l_en.pytorch.d2l import torch as d2l

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap="Reds"):
    """ show heatmaps of matrices.

    :param matrices: torch.shape  (num_rows, num_cols, matrix[0], matrix[1])
    :param xlabel:
    :param ylabel:
    :param titles:
    :param figsize:
    :param cmap: imshow函数的参数，cmap='gray'表示灰度图
    :return:
    """
    d2l.use_svg_display() # svg的分辨率更高
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows-1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    d2l.plt.show()

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1, ), requires_grad=True))

    def forward(self, queries, keys, values):
        """

        :param queries:
        :param keys:
        :param values:
        :return:  shape of the output `queries` and `attention_weights`:
        (no. of queies, no. of key-value pairs)
        """
        # 将query的维度扩张至key的维度
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w) ** 2 / 2, dim=1
        )
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

if __name__=="__main__":
    # attention_weights = torch.eye(10).reshape((1, 1, 10, 10)) # diagonal matrix
    # show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')

    n_train = 50  # NO. of training examples
    x_train, _ = torch.sort(torch.rand(n_train) * 5)  # training inputs

    def f(x):
        return 2 * torch.sin(x) + x ** 0.8

    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # training outputs
    x_test = torch.arange(0, 5, 0.1)  # testing examples
    y_truth = f(x_test)
    n_test = len(x_test)


    # Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
    # same testing inputs (i.e., same queries)
    X_repeat = x_test.repeat_interleave(n_train).reshape(-1, n_train)
    # Note that `x_train` contains the keys. Shape of `attention_weights`:
    # (`n_test`, `n_train`), where each row contains attention weights to be
    # assigned among the values (`y_train`) given each query
    attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
    # Each element of `y_hat` is weighted average of values, where weights are
    # attention weights
    y_hat = torch.matmul(attention_weights, y_train)
    plot_kernel_reg(y_hat)


    # Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
    # same training inputs
    X_tile = x_train.repeat((n_train, 1))
    # Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
    # same training outputs
    Y_tile = y_train.repeat((n_train, 1))
    # Shape of `keys`: ('n_train', 'n_train' - 1)
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)
                            ].reshape((n_train, -1))
    # Shape of `values`: ('n_train', 'n_train' - 1)
    values = Y_tile[ (1 - torch.eye(n_train)).type(torch.bool)
                            ].reshape((n_train, -1))

    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
        animator.add(epoch + 1, float(l.sum()))