# 2 Parameter Management

* 我们有几种方法可以访问、初始化和绑定模型参数。
* 我们可以使用自定义初始化方法。

在选择了架构并设置了超参数后，我们就进入了训练阶段 (train loop)。 此时，我们的$\text{\colorbox{black}{\color{yellow}目标}}$ 是找到使损失函数最小化的模型参数值。 经过训练后，我们将需要使用这些参数来做出未来的预测。 此外，有时我们希望提取参数，以便在其他环境中复用它们， 将模型保存下来，以便它可以在其他软件中执行， 或者为了获得科学的理解而进行检查。

之前的介绍中，我们只依靠深度学习框架来完成训练的工作， 而忽略了操作参数的具体细节。 本节，我们将介绍以下内容：

* Accessing parameters (访问参数) for debugging, diagnostics, and visualizations.
* Sharing parameters across different model components.
* Parameter initialization.

我们首先看一下具有单隐藏层的多层感知机。

```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```

tensor([[0.1552],
[0.1666]], grad_fn=<AddmmBackward0>)

## 2.1 [**Parameter Access**] 参数访问

我们从已有模型中访问参数。 当通过 `Sequential` 类定义模型时， 我们可以 $\color{red}通过索引访问$ 模型的任意层。 这就像模型是一个列表一样，每层的参数都在其属性中。 如下所示，我们可以检查第二个全连接层的参数。

```python
print(net[2].state_dict())
```

OrderedDict([('weight', tensor([[-0.0536,  0.3236,  0.3383,  0.1169, -0.1027,  0.3006, -0.2254, -0.1367]])), ('bias', tensor([0.1964]))])
输出的结果告诉我们一些重要的事情：

- 首先，这个全连接层包含两个参数，分别是该层的权重和偏置。
- 两者都存储为单精度浮点数（float32）。

Note that the names of the parameters (例如 `weight`, `bais`) allow us to $\color{red}\text{uniquely identify}$ each layer's parameters, even in a network containing hundreds of layers.

### 2.1.1 [**Targeted Parameters**] 目标参数

Note that each parameter is represented as $\text{\colorbox{black}{\color{yellow}an instance of}}$ the parameter class. To do anything useful with the parameters, $\color{red}\text{we first need to access the underlying numerical values}$. There are several ways to do this.

下面的代码从第二个全连接层（即第三个神经网络层）提取偏置， 提取后返回的是一个参数类实例，并进一步访问该参数的值。

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

<class 'torch.nn.parameter.Parameter'>
Parameter containing:
tensor([0.1964], requires_grad=True)
tensor([0.1964])
参数是复合的对象，包含值、梯度和额外信息。 这就是我们需要显式参数值的原因。 除了值之外，我们还可以访问每个参数的梯度。 在上面这个网络中，由于我们还没有调用反向传播，所以参数的梯度处于初始状态。

```python
net[2].weight.grad == None
```

True

### 2.1.2 [**All Parameters at Once**] 一次性访问所有参数

当我们需要对所有参数执行操作时，逐个访问它们可能会很麻烦。 当我们处理更复杂的块（例如，嵌套块）时，情况可能会变得特别复杂， 因为我们需要 $\color{red}\text{递归整个树}$ 来提取每个子块的参数。 下面，我们将通过演示来比较访问第一个全连接层的参数和访问所有层。

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
这为我们提供了另一种访问网络参数的方式，如下所示。

```python
net.state_dict()['2.bias'].data
```

tensor([0.1964])

### 2.1.3 Collecting Parameters from Nested Blocks 嵌套块

让我们看看，如果我们将多个块相互嵌套，参数命名约定是如何工作的。 我们首先定义一个生成块的函数（可以说是“块工厂”），然后将这些块组合到更大的块中

```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # Nested here
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

tensor([[0.0870],
[0.0870]], grad_fn=<AddmmBackward0>)
设计了网络后，我们看看它是如何工作的。

```python
print(rgnet)
```

Sequential(
(0): Sequential(
(block 0): Sequential(
(0): Linear(in_features=4, out_features=8, bias=True)
(1): ReLU()
(2): Linear(in_features=8, out_features=4, bias=True)
(3): ReLU()
)
(block 1): Sequential(
(0): Linear(in_features=4, out_features=8, bias=True)
(1): ReLU()
(2): Linear(in_features=8, out_features=4, bias=True)
(3): ReLU()
)
(block 2): Sequential(
(0): Linear(in_features=4, out_features=8, bias=True)
(1): ReLU()
(2): Linear(in_features=8, out_features=4, bias=True)
(3): ReLU()
)
(block 3): Sequential(
(0): Linear(in_features=4, out_features=8, bias=True)
(1): ReLU()
(2): Linear(in_features=8, out_features=4, bias=True)
(3): ReLU()
)
)
(1): Linear(in_features=4, out_features=1, bias=True)
)
因为层是分层嵌套的，所以我们也可以像通过嵌套列表索引一样访问它们。 下面，我们访问第一个主要的块中、第二个子块的第一层的偏置项。

```python
rgnet[0][1][0].bias.data
```

tensor([ 0.2247,  0.3228, -0.2617, -0.0094, -0.2431, -0.3883,  0.1471,  0.4484])

## 2.2 Parameter Initialization

知道了如何访问参数后，现在我们看看如何正确地初始化参数。 我们在 [4.8节](https://zh.d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#sec-numerical-stability)中讨论了良好初始化的必要性。 深度学习框架提供默认随机初始化， 也允许我们创建自定义初始化方法， 满足我们通过其他规则实现初始化权重。

$\color{red}\text{By default}$, PyTorch $\text{\colorbox{black}{\color{yellow}initializes}}$ $\color{magenta}\text{weight}$ and $\color{magenta}\text{bias}$ matrices $\color{red}\text{uniformly}$ by drawing (计算) $\text{\colorbox{black}{\color{yellow}from a range}}$ that is $\text{\colorbox{black}{\color{yellow}computed according}}$ to the $\color{magenta}\text{input}$ and $\color{magenta}\text{output dimension}$.

PyTorch's `nn.init` module provides a variety of preset initialization methods.

### 2.2.1 [**Built-in Initialization**] 内置初始化 (`nn.init.函数_()`)

让我们首先调用内置的初始化器。 下面的代码将所有权重参数初始化为标准差 $0.01$ 的 $高斯随机变量$， 且将偏置参数设置为0。

```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

(tensor([-0.0002, -0.0145, -0.0036, -0.0147]), tensor(0.))
我们还可以将所有参数初始化为 $给定的常数$，比如初始化为1。

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

(tensor([1., 1., 1., 1.]), tensor(0.))
我们还可以对某些块应用不同的初始化方法。

例如，下面我们使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42。

```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

tensor([-0.4059, -0.2534,  0.4267,  0.5348])
tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])

### 2.2.2 [**Custom Initialization**] 自定义初始化

有时，深度学习框架没有提供我们需要的初始化方法。 在下面的例子中，我们使用以下的分布为任意权重参数 $w$ 定义初始化方法：

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U(-10, -5) & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}

$$

Again, we implement a `my_init` function to apply to `net`.

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```
Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])

tensor([[ 0.0000, -9.8794, -5.9144, -7.1713],
[ 0.0000, -6.5578, -0.0000, -0.0000]], grad_fn=<SliceBackward0>)
```

注意，我们始终可以直接设置参数。

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

tensor([42.0000, -8.8794, -4.9144, -6.1713])

## 2.3 [**Tied Parameters**] 参数绑定

有时我们希望在多个层间 $\color{red}共享参数$： 我们可以定义一个Dense Layer，然后使用它的参数来设置另一个层的参数。

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

tensor([True, True, True, True, True, True, True, True])
tensor([True, True, True, True, True, True, True, True])
这个例子表明第三个和第五个神经网络层的参数是绑定的。 它们不仅值相等，而且由相同的张量表示。 因此，如果我们改变其中一个参数，另一个参数也会改变。

- 你可能会思考：当参数绑定时，梯度会发生什么情况？
- 答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层 （即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。

## Summary

* We have several ways to access, initialize, and tie model parameters.
* We can use custom initialization.

## Exercises

1. Use the `FancyMLP` model defined in :numref:`sec_model_construction` and access the parameters of the various layers.
2. Look at the initialization module document to explore different initializers.
3. Construct an MLP containing a shared parameter layer and train it. During the training process, observe the model parameters and gradients of each layer.
4. Why is sharing parameters a good idea?

[Discussions](https://discuss.d2l.ai/t/57)
