# 6 卷积

上节我们解析了卷积层的原理，现在我们看看它的实际应用。由于卷积神经网络的设计是用于探索图像数据，本节我们将以图像为例。

## 6.2.1. 互相关运算

严格来说，卷积层是个错误的叫法，因为它所表达的运算其实是 *互相关运算* （cross-correlation），而不是卷积运算。 根据 [6.1节](https://zh.d2l.ai/chapter_convolutional-neural-networks/why-conv.html#sec-why-conv)中的描述，在卷积层中，输入张量和核张量通过互相关运算产生输出张量。

首先，我们暂时忽略通道（第三维）这一情况，看看如何处理二维图像数据和隐藏表示。在 [图6.2.1](https://zh.d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#fig-correlation)中，输入是高度为3、宽度为3的二维张量（即形状为3×3）。卷积核的高度和宽度都是2，而卷积核窗口（或卷积窗口）的形状由内核的高度和宽度决定（即2×2）。

![../_images/correlation.svg](https://zh.d2l.ai/_images/correlation.svg)

```python
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
```

## 6.4.1. 多输入通道

在 [图6.4.1](https://zh.d2l.ai/chapter_convolutional-neural-networks/channels.html#fig-conv-multi-in)中，我们演示了一个具有两个输入通道的二维互相关运算的示例。阴影部分是第一个输出元素以及用于计算这个输出的输入和核张量元素：(1×1+2×2+4×3+5×4)+(0×0+1×1+3×2+4×3)=56。

![../_images/conv-multi-in.svg](https://zh.d2l.ai/_images/conv-multi-in.svg)

简而言之，我们所做的就是对每个通道执行互相关操作，然后将结果相加。

```python
import torch
from d2l import torch as d2l

def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

## 6.4.2. 多输出通道

到目前为止，不论有多少输入通道，我们还只有一个输出通道。然而，正如我们在 [6.1.4.1节](https://zh.d2l.ai/chapter_convolutional-neural-networks/why-conv.html#subsec-why-conv-channels)中所讨论的，每一层有多个输出通道是至关重要的。

- 在最流行的神经网络架构中，随着神经网络层数的加深，我们常会增加输出通道的维数，通过减少空间分辨率以获得更大的通道深度。

直观地说，我们可以将每个通道看作是对不同特征的响应。而现实可能更为复杂一些，因为每个通道不是独立学习的，而是为了共同使用而优化的。因此，多输出通道并不仅是学习多个单通道的检测器。

用 $c_i$ 和 $c_o$ 分别表示输入和输出通道的数目，并让 $k_h$ 和 $k_w$ 为卷积核的高度和宽度。为了获得多个通道的输出，我们可以为 $\color{red}每个输出通道$ 创建一个形状为 $c_i×k_h×k_w$ 的卷积核张量，这样卷积核的形状是 ${\color{red}c_o}×c_i×k_h×k_w$。在互相关运算中，每个输出通道先获取所有输入通道，再以对应该输出通道的卷积核计算出结果。

```python
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```

## 6.4.3. 1×1 卷积层

* 多输入多输出通道可以用来扩展卷积层的模型。
* 当以每像素为基础应用时，1×1卷积层相当于全连接层。
* 1×1卷积层通常用于调整网络层的通道数量和控制模型复杂性。

$1×1$ 卷积，即 $k_h=k_w=1$，看起来似乎没有多大意义。 毕竟，卷积的本质是有效提取相邻像素间的相关特征，而 $1×1$ 卷积显然没有此作用。 尽管如此，$1×1$ 仍然十分流行，经常包含在复杂深层网络的设计中。下面，让我们详细地解读一下它的实际作用。

因为使用了最小窗口，$1×1$ 卷积失去了卷积层的特有能力——在高度和宽度维度上，识别相邻元素间相互作用的能力。 其实 $1×1$ 卷积的 $\color{red}唯一计算发生在通道上$。

[图6.4.2](https://zh.d2l.ai/chapter_convolutional-neural-networks/channels.html#fig-conv-1x1)展示了

- 使用 $1×1$ 卷积核与 $3$ 个 $\color{magenta}输入$ 通道和 $2$ 个 $\color{red}输出$ 通道的互相关计算。
- 这里输入和输出具有相同的高度和宽度，$\color{red}输出中的每个元素$ 都是从$\color{magenta}输入图像$中$\text{\color{yellow}\colorbox{black}{同一位置}}$的元素的$\text{\color{yellow}\colorbox{black}{线性组合}}$。

我们可以将 $1×1$ 卷积层看作是在 每个 $\text{\color{yellow}\colorbox{black}{像素位置应用的全连接层}}$，以 $\color{magenta}c_i$ 个输入值转换为 $\color{red}c_o$ 个输出值。 因为这仍然是一个卷积层，所以跨像素的权重是一致的。 同时，$1×1$ 卷积层需要的权重维度为$c_o×c_i$，再额外加上一个偏置。

![../_images/conv-1x1.svg](https://zh.d2l.ai/_images/conv-1x1.svg)

```python
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
```
