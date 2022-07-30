# 1 Layers and Blocks

:label:`sec_model_construction`


* 一个块可以由许多层组成；一个块可以由许多块组成。
* 块可以包含代码。
* 块负责大量的内部处理，包括参数初始化和反向传播。
* 层和块的顺序连接由`Sequential`块处理。


之前首次介绍神经网络时，我们关注的是具有 $\text{\colorbox{black}{\color{yellow}单一输出}}$ 的线性模型。 在这里，整个模型只有一个输出。 注意，单个神经网络

- （1）接受一些输入；
- （2）生成相应的标量输出；
- （3）具有一组相关  *参数* （parameters），更新这些参数可以优化某目标函数。

然后，当考虑具有 $\text{\colorbox{black}{\color{yellow}多个输出}}$ 的网络时， 我们利用 $\text{\colorbox{black}{\color{red}矢量化算法}}$ 来描述整层神经元。 像单个神经元一样，$\text{\colorbox{black}{\color{red}Layer}}$

- （1）接受 $\text{\colorbox{black}{\color{yellow}一组}}$ 输入，
- （2）生成相应的 $\text{\colorbox{black}{\color{yellow}一组}}$ 输出，
- （3）由 $\text{\colorbox{black}{\color{yellow}一组}}$ 可调整参数描述。

当我们使用softmax回归时，一个单层本身就是模型。 然而，即使我们随后引入了多层感知机，我们仍然可以认为该模型保留了上面所说的基本架构。

对于 MLPs 而言，整个模型及其组成层都是这种架构。 整个模型接受原始输入（特征），生成输出（预测）， 并包含一些参数（所有组成层的参数集合）。 同样，每个单独的层接收输入（由前一层提供）， 生成输出（到下一层的输入），并且具有一组可调参数， 这些参数根据从下一层反向传播的信号进行更新。

事实证明，研究讨论 “比单个层大” 但 “比整个模型小” 的 $\color{red}组件 更有价值$。 例如，在计算机视觉中广泛流行的ResNet-152架构就有数百层， 这些层是由 *层组* （groups of layers）的重复模式组成。 这个ResNet架构赢得了2015年ImageNet和COCO计算机视觉比赛 的识别和检测任务 [[He et al., 2016a]](https://zh.d2l.ai/chapter_references/zreferences.html#he-zhang-ren-ea-2016)。 目前ResNet架构仍然是许多视觉任务的首选架构。 在其他的领域，如自然语言处理和语音， 层组以各种重复模式排列的类似架构现在也是普遍存在。

为了实现这些复杂的网络，我们引入了神经网络 $\color{red}block$ 的概念。  *块* （block）可以描述单个层、由多个层组成的组件或整个模型本身。 使用块进行抽象的一个好处是可以将一些块组合成更大的组件， 这一过程通常是递归的，如 [图5.1.1](https://zh.d2l.ai/chapter_deep-learning-computation/model-construction.html#fig-blocks)所示。 通过定义代码来按需生成任意复杂度的块， 我们可以通过简洁的代码实现复杂的神经网络。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://zh.d2l.ai/_images/blocks.svg" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图5.1.1 多个层被组合成块，形成更大的模型¶
  	</div>
</center>

从编程的角度来看，$\text{\colorbox{black}{\color{red}块}}$ 由 $\text{\colorbox{black}{\color{red}类 （class）}}$表示。 它的任何 子类 都必须定义一个将其输入转换为输出的 $\text{\colorbox{black}{\color{red}前向传播函数}}$， 并且必须存储任何必需的参数。 注意，有些块不需要任何参数。 最后，为了计算梯度，块必须具有反向传播函数。 在定义我们自己的块时，由于 auto differentiation（在 [2.5节](../C2_preliminaries/2_5_autograd.md) 中引入） 提供了一些后端实现，我们 $\text{\colorbox{black}{\color{yellow}只需要考虑}} $ $\color{red}前向传播函数$ 和 $\color{red}必需的参数$。

在构造自定义块之前，我们先回顾一下多层感知机 （ [4.3节](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp-concise.html#sec-mlp-concise) ）的代码。 下面的代码生成一个网络，其中包含一个具有256个单元和ReLU激活函数的全连接隐藏层， 然后是一个具有10个隐藏单元且不带激活函数的全连接输出层。

```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

tensor([[-0.0275,  0.0889, -0.1878, -0.0165,  0.0356, -0.3514, -0.0725,  0.0725,
0.0508, -0.0560],
[ 0.0224,  0.0984, -0.2499,  0.0313,  0.1201, -0.5500, -0.1515,  0.1012,
0.0086, -0.0528]], grad_fn=<AddmmBackward0>)
In this example, we constructed our model by instantiating an `nn.Sequential`, with $\text{\colorbox{black}{\color{red}layers in the order}}$ that they should be executed passed as arguments (层的执行顺序是作为参数传递的). In short,  **`nn.Sequential` defines a special kind of `Module`**, the class that presents a $\text{\colorbox{black}{\color{red}block}}$ in PyTorch. It maintains an $\text{\colorbox{black}{\color{red}ordered list 有序列表}}$ of constituent `Module`s. Note that each of the two fully-connected layers is an instance of the `Linear` class which is itself a subclass of `Module`.

The forward propagation (`forward`) function is also remarkably simple: it chains each block in the list together, passing the output of each as the input to the next. Note that until now, we have been invoking our models via the construction `net(X)` to obtain their outputs. This is actually just shorthand for `net.__call__(X)`.

## 1.1 [**A Custom Block**] 自定义块

要想直观地了解 Block 是如何工作的，最简单的方法就是自己实现一个。 在实现我们自定义块之前，我们简要总结一下每个块必须提供的基本功能：

1. 将输入数据 `X` 作为其前向传播函数 `def forward(self, X, *args)` 的参数。
2. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是 $\text{\colorbox{black}{\color{red}自动发生}}$ （torch内部实现了）的。
4. 存储和访问前向传播计算所需的参数。
5. 根据需要 $\text{\colorbox{black}{\color{red}初始化模型参数,optional}}$。

In the following snippet 代码片段, we code up a block from scratch corresponding to an MLP with one hidden layer with 256 hidden units, and a 10-dimensional output layer. Note that the `MLP` class below inherits the class that represents a block. We will heavily rely on the parent class's functions, supplying only our own constructor (the `__init__` function in Python) and the forward propagation function `forward`.

```python
class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Module` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # Hidden layer
        self.out = nn.Linear(256, 10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(X)))
```

我们首先看一下前向传播函数，它以`X` 作为输入， 计算带有激活函数的隐藏表示，并输出其未规范化的输出值。 在这个`MLP` 实现中，两个层都是实例变量。 要了解这为什么是合理的，可以想象实例化两个多层感知机（`net1`和`net2`）， 并根据不同的数据对它们进行训练。 当然，我们希望它们学到两种不同的模型。

接着我们实例化多层感知机的层，然后在每次调用前向传播函数时调用这些层。 注意一些关键细节：

- First, our customized `__init__` function invokes the $\text{\colorbox{black}{\color{red}parent class's}}$ `__init__` function
  via `super().__init__()` 省去了重复编写模版代码的痛苦.
- We then instantiate our two fully-connected layers, assigning them to `self.hidden` and `self.out`.

注意，除非我们实现一个新的运算符， 否则我们 $\text{\colorbox{black}{\color{yellow}不必担心}}$ 反向传播函数或参数初始化， 系统将自动生成这些。

```python
net = MLP()
net(X)
```

tensor([[ 0.1298,  0.2597, -0.0968, -0.0494,  0.1476,  0.1454,  0.1632,  0.0199,
-0.0087,  0.3480],
[ 0.1466,  0.1623, -0.0858, -0.0544,  0.1312,  0.2089,  0.1057,  0.0484,
-0.0889,  0.3298]], grad_fn=<AddmmBackward0>)
$\text{\colorbox{black}{\color{red}block}}$ 的一个主要优点是它的多功能性。 我们可以子类化块以创建层（如全连接层的类）、 整个模型（如上面的`MLP` 类）或具有中等复杂度的各种组件。 我们在接下来的章节中充分利用了这种多功能性， 比如在处理卷积神经网络时。

## 1.2 [**The Sequential Block**]

现在我们可以更仔细地看看`Sequential`类是如何工作的， 回想一下`Sequential`的设计是为了把其他模块串起来。 为了构建我们自己的简化的`MySequential`， 我们只需要定义两个关键函数：

1. A function to append blocks one by one to a `list`.
2. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。

The following `MySequential` class delivers the same functionality of the default `Sequential` class.

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # Here, `module` is an instance of a `Module` subclass. We save it
            # in the member variable `_modules` of the `Module` class, and its
            # type is OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._modules.values():
            X = block(X)
        return X
```

`__init__` 函数将每个模块逐个添加到 OrderdDict `_modules` 中。 你可能会好奇为什么每个 `Module` 都有一个 `_modules` 属性？ 以及为什么我们使用它而不是自己定义一个 Python列表？ 简而言之，`_modules` 的 $\text{\colorbox{black}{\color{red}主要优点}}$ 是： 在模块的参数初始化过程中， 系统知道在 `_modules` 字典中查找需要初始化参数的子块。

当`MySequential`的前向传播函数被调用时， 每个添加的块都按照它们被添加的顺序执行。 现在可以使用我们的`MySequential`类重新实现多层感知机。

```python
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

tensor([[-0.2045, -0.1206,  0.2077,  0.0615, -0.1246,  0.0457, -0.0679, -0.3117,
0.0269, -0.3475],
[-0.2765, -0.2260,  0.1733,  0.1343, -0.3023,  0.0324, -0.1404, -0.4459,
-0.0769, -0.2924]], grad_fn=<AddmmBackward0>)
请注意，`MySequential` 的用法与之前为 `Sequential` 类编写的代码相同 （如 [4.3节](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp-concise.html#sec-mlp-concise) 中所述）。

## 1.3 [**Executing Code in the Forward Propagation Function**]

The `Sequential` 类使模型构造变得简单， 允许我们组合新的架构，而不必定义自己的类。 然而，并不是所有的架构都是简单的顺序架构。 当需要更强的灵活性时，我们需要定义自己的块。 例如，我们可能希望在前向传播函数中执行 Python 的控制流。 此外，我们可能希望执行任意的数学运算， 而不是简单地依赖预定义的神经网络层。

到目前为止， 我们网络中的所有操作都对网络的激活值及网络的参数起作用。 $\color{red}然而$，有时我们可能希望 $\text{\colorbox{black}{\color{yellow}合并}}$ 既不是上一层的结果也不是可更新参数的项， 我们称之为 $\text{\colorbox{black}{\color{red}常数参数}}$ （constant parameter）。

- 例如，我们需要一个计算函数 $f(\mathbf{x},\mathbf{w}) = {\color{red}c} \cdot \mathbf{w}^\top \mathbf{x}$ 的层，其中 $x$ 是输入， $w$ 是参数， $\color{red}c$ 是某个在优化过程中 $\color{red}没有更新的指定常量$。 因此我们实现了一个 `FixedHiddenMLP` 类，如下所示：
- 需要
  - 1） 自定义常数参数，`self.rand_weight`
  - 2）然后调用 torch.nn 里面的 functioinal 库 中的函数，
  - 3）实现相应运算。

```python
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
```

In this `FixedHiddenMLP` model, 我们实现了一个隐藏层， 其权重 (`self.rand_weight`) 在实例化时被随机初始化，之后为常量。这个权重不是一个模型参数，因此它永远不会被反向传播更新。 然后，神经网络将这个固定层的输出通过一个全连接层。

注意，在返回输出之前，模型做了一些不寻常的事情：

- 它运行了一个while循环，$L_1$ 范数大于 $1$ 的条件下， 将输出向量除以 $2$，直到它满足条件为止。
- 最后，模型返回了 `X` 中所有项的和。
- 注意，此操作可能不会常用于在任何实际任务中， 我们只是向你展示如何将任意代码集成到神经网络计算的流程中。

```python
net = FixedHiddenMLP()
net(X)
```

```
tensor(0.1288, grad_fn=<SumBackward0>)
```

我们可以混合搭配各种组合块的方法。 在下面的例子中，我们以一些想到的方法嵌套块。

```python
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

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

```
tensor(-0.2879, grad_fn=<SumBackward0>)
```

## 1.4 Efficiency

你可能会开始担心操作效率的问题。 毕竟，我们在一个高性能的深度学习库中进行了大量的字典查找、 代码执行和许多其他的Python代码。 Python的问题 [全局解释器锁 (GIL)](https://wiki.python.org/moin/GlobalInterpreterLock) 是众所周知的。 在深度学习环境中，我们担心速度极快的 GPU 可能要等到 CPU 运行 Python 代码后才能运行另一个作业。

The best way to speed up Python is by avoiding it altogether （$\color{red}避免同时使用 cpu、gpu$）。 Gluon 这样做的一个方法是允许 *$\color{red}混合式编程$* （hybridization），这将在后面描述。

Here, the Python interpreter executes a block the first time it is invoked. The Gluon runtime records what is happening and the next time around it short-circuits calls to Python.

在某些情况下，这可以大大加快运行速度， 但当控制流（如上所述）在不同的网络通路上引导不同的分支时，需要格外小心。 我们建议感兴趣的读者在读完本章后，阅读混合式编程部分（ [12.1节](https://zh.d2l.ai/chapter_computational-performance/hybridize.html#sec-hybridize) ）来了解编译。

## Summary

* Layers are blocks.
* Many layers can comprise a block.
* Many blocks can comprise a block.
* A block can contain code.
* Blocks take care of lots of housekeeping, including parameter initialization and backpropagation.
* Sequential concatenations of layers and blocks are handled by the `Sequential` block.

## Exercises

1. What kinds of problems will occur if you change `MySequential` to store blocks in a Python list?
2. Implement a block that takes two blocks as an argument, say `net1` and `net2` and returns the concatenated output of both networks in the forward propagation. This is also called a parallel block.
3. Assume that you want to concatenate multiple instances of the same network. Implement a factory function that generates multiple instances of the same block and build a larger network from it.

[Discussions](https://discuss.d2l.ai/t/55)
