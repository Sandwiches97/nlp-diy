# 3 Concise Implementation of Linear Regression

:label:`sec_linear_concise`

* 我们可以使用PyTorch的高级API更简洁地实现模型。
* 在PyTorch中，`data`模块提供了数据处理工具，`nn` 模块定义了大量的神经网络层和常见损失函数。
* 我们可以通过 `_` 结尾的方法将参数替换，从而初始化参数。


在过去的几年里，出于对深度学习强烈的兴趣， 许多公司、学者和业余爱好者开发了各种成熟的开源框架。 这些框架可以自动化基于梯度的学习算法中重复性的工作。 在 [3_2_linear_regression_scratch.md](3_2_linear_regression_scratch.md) 中，我们只运用了： （1）通过张量来进行数据存储和线性代数； （2）通过自动微分来计算梯度。 实际上，由于数据迭代器、损失函数、优化器和神经网络层很常用， 现代深度学习库也为我们实现了这些组件。

在本节中，我们将介绍如何通过使用深度学习框架来简洁地实现 [3_2_linear_regression_scratch.md](3_2_linear_regression_scratch.md) 中的线性回归模型。

## 3.1 Generating the Dataset

To start, we will generate the same dataset as in [3_2_linear_regression_scratch.md](3_2_linear_regression_scratch.md). 

```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
```

```python
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## 3.2 Reading the Dataset

Rather than rolling our own iterator, we can [**call upon $\color{red}\text{the existing API}$ `from torch.utils.data import DataLoader` in a framework to read data.**] We pass in `features`and`labels`as arguments and specify`batch_size`when instantiating a data iterator object. Besides, the boolean value`is_train` indicates whether or not we want the data iterator object to shuffle the data on each epoch (pass through the dataset).

```python
def load_array(data_arrays: Tuple[list, list], batch_size, is_train=True):
    """ construct a pytorch iterator """
    dataset = data.TensorDataset(*data_arrays) # 类似 zip函数
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```python
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

Now we can use `data_iter` in much the same way as we called the `data_iter` function in [3_2_linear_regression_scratch.md](3_2_linear_regression_scratch.md).

To verify that it is working, we can read and print the first minibatch of examples. Comparing with [3_2_linear_regression_scratch.md](3_2_linear_regression_scratch.md), here we use `iter` to construct a Python iterator and use `next` to obtain the first item from the iterator.

```python
next(iter(data_iter))
```

[tensor([[-0.0385, -1.5909],
         [-0.1432,  0.9797],
         [-1.2551,  0.1154],
         [ 2.0345,  0.3657],
         [ 0.0179,  0.4323],
         [-1.9013, -0.6234],
         [-1.7319,  0.3589],
         [-0.2995,  0.0825],
         [-1.7680, -1.3965],
         [-0.8613, -0.5467]]),
 tensor([[ 9.5418],
         [ 0.5854],
         [ 1.3041],
         [ 7.0231],
         [ 2.7599],
         [ 2.5210],
         [-0.4856],
         [ 3.3295],
         [ 5.4176],
         [ 4.3249]])]
## 3.3 Defining the Model

当我们在 [3.2节](3_2_linear_regression_scratch.md) 中实现线性回归时， 我们明确定义了模型参数变量，并编写了计算的代码，这样通过基本的线性代数运算得到输出。 但是，如果模型变得更加复杂，且当你几乎每天都需要实现模型时，你会想简化这个过程。 这种情况类似于为自己的博客从零开始编写网页。 做一两次是有益的，但如果每个新博客你就花一个月的时间重新开始编写网页，那并不高效。

对于标准深度学习模型，我们可以使用框架的预定义好的层。这使我们只需关注使用哪些层来构造模型，而不必关注层的实现细节。
We will first define a model variable `net`, which will refer to an instance (实例) of the `Sequential` class.

The `Sequential` class defines a container for several layers that will be $\color{red}\text{chained 串联}$ together. Given input data, a `Sequential` instance passes it through the first layer, in turn passing the output
as the second layer's input and so forth.

In the following example, our model consists of only one layer, so we do not really need `Sequential`. But since nearly all of our future models will involve multiple layers, we will use it anyway just to familiarize you with the most standard workflow.

回顾 [图3.1.2](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#fig-single-neuron) 中的单层网络架构， 这一单层被称为 *全连接层* （fully-connected layer）， 因为它的每一个输入都通过矩阵-向量乘法得到它的每个输出。

在PyTorch中，全连接层在`Linear`类中定义。 值得注意的是，我们将两个参数传递到`nn.Linear`中。 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。

```python
# `nn` is an abbreviation for neural networks
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
```
## Initializing Model Parameters

Before using `net`,  我们需要初始化模型参数。 如在线性回归模型中的权重和偏置。 深度学习框架通常有预定义的方法来初始化参数。 在这里，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样， 偏置参数将初始化为零。

As we have specified the input and output dimensions when constructing `nn.Linear`, now we can access the parameters directly to specify their initial values.

- We first locate the layer by `net[0]`, which is the first layer in the network, and
- then use the `weight.data` and `bias.data` methods to access the parameters.
- Next we use the replace methods `normal_` and `fill_` to overwrite parameter values.

```python
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```
tensor([0.])
## 3.4 Defining the Loss Function

[计算均方误差使用的是`MSELoss`类，也称为平方L2范数。 默认情况下，它返回所有样本损失的平均值。

```python
loss = nn.MSELoss()
```
## Defining the Optimization Algorithm

Minibatch stochastic gradient descent is a standard tool for optimizing neural networks
and thus PyTorch supports it alongside a number of variations on this algorithm in the `optim` module.
When we **instantiate an `SGD` instance,** we will specify 

- the parameters $\color{red}待优化参数$ to optimize over (obtainable from our net via `net.parameters()`), with a dictionary of hyperparameters required by our optimization algorithm.

Minibatch stochastic gradient descent just requires that we set the value `lr`, which is set to 0.03 here.

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```
## 3.7 Training

通过深度学习框架的高级API来实现我们的模型只需要相对较少的代码。 我们不必单独分配参数、不必定义我们的损失函数，也不必手动实现小批量随机梯度下降。 当我们需要更复杂的模型时，高级API的优势将大大增加。 当我们有了所有的基本组件，训练过程代码与我们从零开始实现时所做的非常相似。

回顾一下：在每个迭代周期里，我们将完整遍历一次数据集 (`train_data`), 不停地从中获取一个小批量的输入和相应的标签。 对于每一个小批量，我们会进行以下步骤:

1. 通过调用`net(X)`生成预测（前向传播）
2. 计算损失函数 `l`
3. 通过损失函数`l`进行反向传播来计算梯度
4. 通过调用优化器来更新模型参数。

For good measure, we compute the loss after each epoch and print it to monitor progress.

```python
num_epoch = 3
for epoch in range(num_epoch):
    for X, y in data_iter:
        y_hat = net(X)      # 1
        l = loss(y_hat, y)  # 2
        trainer.zero_grad() # 3 后向传播前，需要将grad清零
        l.backward()        # 3
        trainer.step()      # 4
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```
```
epoch 1, loss 0.000207
epoch 2, loss 0.000105
epoch 3, loss 0.000105
```

下面我们比较生成数据集的真实参数和通过有限数据训练获得的模型参数。 要访问参数，我们首先从`net`访问所需的层，然后读取该层的权重和偏置。 正如在从零开始实现中一样，我们估计得到的参数与生成数据的真实参数非常接近。

```python
w = net[0].weight.data
print('error in estimating w:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
```
error in estimating w: tensor([-0.0005, -0.0003])
error in estimating b: tensor([-0.0002])
## Summary

* Using PyTorch's high-level APIs, we can implement models much more concisely.
* In PyTorch, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers and common loss functions.
* We can initialize the parameters by replacing their values with methods ending with `_`.

## Exercises

1. If we replace `nn.MSELoss(reduction='sum')` with `nn.MSELoss()`, how can we change the learning rate for the code to behave identically. Why?
2. Review the PyTorch documentation to see what loss functions and initialization methods are provided. Replace the loss by Huber's loss.
3. How do you access the gradient of `net[0].weight`?

[Discussions](https://discuss.d2l.ai/t/45)
