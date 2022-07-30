# 4 Custom Layers


* 我们可以通过基本层类设计自定义层。这允许我们定义灵活的新层，其行为与深度学习框架中的任何现有层不同。
* 在自定义层定义完成后，我们就可以在任意环境和网络架构中调用该自定义层。
* 层可以有局部参数，这些参数可以通过内置函数创建。



深度学习成功背后的一个因素是神经网络的灵活性： 我们可以用创造性的方式组合不同的层，从而设计出适用于各种任务的架构。 例如，研究人员发明了专门用于处理图像、文本、序列数据和执行动态规划的层。 未来，你会遇到或要自己发明一个现在在深度学习框架中还不存在的层。 在这些情况下，你必须构建自定义层。在本节中，我们将向你展示如何构建。

## 4.1 (**Layers without Parameters**)

首先，我们构造一个没有任何参数的自定义层。 如果你还记得我们在 [5_1_model_construction.md](5_1_model_construction.md) 对 block 的介绍， 这应该看起来很眼熟。下面的 `CenteredLayer` 类要从其输入中减去均值。 要构建它，我们只需继承 the base layer class 并实现前向传播功能。

```python
import torch
from torch import nn
from torch.nn import functional as F


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

让我们向该层提供一些数据，验证它是否能按预期工作。

```python
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

tensor([-2., -1.,  0.,  1.,  2.])
We can now [$\text{\colorbox{black}{\color{yellow}incorporate}}$ our layer as a component $\text{\colorbox{black}{\color{yellow}in}}$ constructing more complex models.]

```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```
As an extra sanity check (健全性检查), 我们可以在向该网络发送随机数据后，检查均值是否为0。 由于我们处理的是浮点数，因为存储精度的原因，我们仍然可能会看到一个非常小的非零数。

```python
Y = net(torch.rand(4, 8))
Y.mean()
```
tensor(5.5879e-09, grad_fn=<MeanBackward0>)
## 4.2 [**Layers with Parameters**]

以上我们知道了如何定义简单的层，下面我们继续定义 $\text{\colorbox{black}{\color{red}具有参数的层}}$， 这些参数可以通过训练进行调整。 我们可以使用 $\text{\colorbox{black}{\color{red}内置函数}}$ 来创建参数，这些函数提供一些基本的管理功能。 比如

- 管理访问、
- 初始化、
- 共享、
- 保存和
- 加载模型参数。

这样做的好处之一是：我们不需要为每个自定义层编写自定义的序列化程序。

现在，让我们实现自定义版本的全连接层。 回想一下，该层需要两个参数，

- 一个用于表示 weight，
- 另一个用于表示 bais。

在此实现中，我们使用修正线性单元 (ReLU) 作为激活函数。 该层需要输入参数： `in_units` and `units`, 分别表示输入数和输出数。

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```
Next, we instantiate the `MyLinear` class and access its model parameters.

```python
linear = MyLinear(5, 3)
linear.weight
```
Parameter containing:
tensor([[-1.4168, -0.5379,  0.4806],
        [ 1.3277, -0.2154, -1.8320],
        [-0.0442, -1.7380,  0.1403],
        [ 0.7403,  0.1926, -1.2225],
        [-0.5699, -0.3245, -0.0856]], requires_grad=True)
我们可以使用自定义层直接执行前向传播计算。

```python
linear(torch.rand(2, 5))
```
tensor([[0.7634, 0.0000, 0.0000],
        [1.1143, 0.0000, 0.0000]])
我们还可以使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层。

```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```
tensor([[2.6498],
        [0.2862]])
## Summary

* We can design custom layers via the basic layer class. This allows us to define flexible new layers that behave differently from any existing layers in the library.
* Once defined, custom layers can be invoked in arbitrary contexts and architectures.
* Layers can have local parameters, which can be created through built-in functions.

## Exercises

1. Design a layer that takes an input and computes a tensor reduction,
   i.e., it returns $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
2. Design a layer that returns the leading half of the Fourier coefficients of the data.

[Discussions](https://discuss.d2l.ai/t/59)
