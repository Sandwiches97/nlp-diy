# 7.5. Batch Normalization

训练深层神经网络是十分困难的，特别是在较短的时间内使他们收敛更加棘手。 在本节中，我们将介绍 *批量规范化* （batch normalization） [[Ioffe &amp; Szegedy, 2015]](https://zh.d2l.ai/chapter_references/zreferences.html#ioffe-szegedy-2015)，这是一种流行且有效的技术，可持续加速深层网络的收敛速度。 再结合在 [7.6节](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html#sec-resnet)中将介绍的残差块，批量规范化使得研究人员能够训练100层以上的网络。


## 7.5.1. 训练深层网络

为什么需要批量规范化层呢？让我们来回顾一下训练神经网络时出现的一些实际挑战。

- 首先，$\color{red}数据预处理的方式通常会对最终结果产生巨大影响$。 回想一下我们应用多层感知机来预测房价的例子（ [4.10节](https://zh.d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html#sec-kaggle-house)）。 使用真实数据时，我们的第一步是标准化输入特征，使其平均值为0，方差为1。 直观地说，这种标准化可以很好地与我们的优化器配合使用，因为它可以将参数的量级进行统一。
- 第二，对于典型的多层感知机或卷积神经网络。当我们训练时，中间层中的变量（例如，多层感知机中的 $\text{\color{yellow}\colorbox{black}{仿射变换输出}}$）可能具有 $\text{\color{red}\colorbox{black}{更广的变化范围}}$：不论是沿着从输入到输出的层，跨同一层中的单元，或是随着时间的推移，模型参数的随着训练更新变幻莫测。 批量规范化的发明者非正式地假设，这些变量分布中的这种偏移可能会阻碍网络的收敛。 直观地说，我们可能会猜想，如果一个层的可变值是另一层的 100倍，这可能需要对学习率进行补偿调整。
- 第三，$\color{red}更深层的网络很复杂，容易过拟合$。 这意味着正则化变得更加重要。

BN 应用于单个可选层（也可以应用到所有层），其原理如下：

- 在每次训练迭代中，我们首先规范化输入，即通过减去其均值并除以其标准差，其中两者均基于当前小批量处理。
- 接下来，我们应用比例系数和比例偏移。 正是由于这个基于 **batch** 统计的 **normalizatoin** ，才有了 *BN* 的名称。

请注意，如果我们尝试使用 batch size = 1 应用批量规范化，我们将无法学到任何东西。 这是因为在减去均值之后，每个隐藏单元将为0。 所以，只有使用$\color{red}足够大的\ batch\ size$，批量规范化这种方法才是有效且稳定的。 请注意，在应用 BN 方法时，$\color{red}batch\ size\ 的选择 (很重要)$可能比没有批量规范化时更重要。

从形式上来说，用 $x∈\mathcal{B}$ 表示一个来自小批量 $\mathcal{B}$ 的输入，批量规范化 BN 根据以下表达式转换$x$：

$$
\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.

$$

在 [(7.5.1)]() 中，$\hat{\boldsymbol{\mu}}_\mathcal{B}$ 是小批量 $\mathcal{B}$ 的样本均值，$\hat{\sigma_\mathcal{B}}$ 是小批量 $\mathcal{B}$ 的样本标准差。 应用 normalization 后，生成的小批量的平均值为 0 和单位方差为 1。 由于单位方差（与其他一些魔法数）是一个主观的选择，因此我们通常包含  *拉伸参数* （scale）$γ$ 和 *偏移参数* （shift）$β$，它们的形状与 $x$ 相同。 请注意，$γ$ 和 $β$ 是需要与其他模型参数一起学习的参数。


由于在训练过程中，中间层的变化幅度不能过于剧烈，而批量规范化将每一层主动居中，并将它们重新调整为给定的平均值和大小（通过 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ 和 $\hat{\sigma_\mathcal{B}}$）。

从形式上来看，我们计算出 [(7.5.1)]() 中的 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ 和 $\hat{\sigma_\mathcal{B}}$，如下所示：

$$
\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}

$$

请注意，我们在方差估计值中添加一个小的常量 $ϵ>0$，以确保我们永远不会尝试除以零，即使在经验方差估计值可能消失的情况下也是如此。估计值 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ 和 $\hat{\sigma_\mathcal{B}}$ 通过使用平均值和方差的噪声（noise）估计来抵消缩放问题。 你可能会认为这种噪声是一个问题，而事实上它是有益的。

事实证明，这是深度学习中一个反复出现的主题。 由于尚未在理论上明确的原因，优化中的各种噪声源通常会导致更快的训练和较少的过拟合：这种变化似乎是正则化的一种形式。 在一些初步研究中， [[Teye et al., 2018]](https://zh.d2l.ai/chapter_references/zreferences.html#teye-azizpour-smith-2018)和 [[Luo et al., 2018]](https://zh.d2l.ai/chapter_references/zreferences.html#luo-wang-shao-ea-2018)分别将批量规范化的性质与贝叶斯先验相关联。 这些理论揭示了为什么批量规范化最适应50∼100范围中的中等批量大小的难题。

另外，批量规范化层在”训练模式“（通过小批量统计数据规范化）和“预测模式”（通过数据集统计规范化）中的功能不同。 在训练过程中，我们无法得知使用整个数据集来估计平均值和方差，所以只能根据每个小批次的平均值和方差不断训练模型。 而在预测模式下，可以根据整个数据集精确计算批量规范化所需的平均值和方差。

现在，我们了解一下批量规范化在实践中是如何工作的。



## 7.5.2. 批量规范化层

回想一下，批量规范化和其他层之间的一个关键区别是，由于批量规范化在完整的小批量上运行，因此我们不能像以前在引入其他层时那样忽略批量大小。 我们在下面讨论这两种情况：全连接层和卷积层，他们的批量规范化实现略有不同。

### 7.5.2.1. 全连接层

通常，我们将 BN layer $\text{\color{yellow}\colorbox{black}{置于}}$全连接层中的 $\text{\color{red}\colorbox{black}{仿射变换}}$和 $\text{\color{magenta}\colorbox{black}{激活函数}}$ $\text{\color{yellow}\colorbox{black}{之间}}$。 设全连接层的输入为 $u$，权重参数和偏置参数分别为 $W$ 和 $b$，激活函数为 $ϕ$，批量规范化的运算符为 BN。 那么，使用批量规范化的全连接层的输出的计算详情如下：

$$
\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).

$$


### 7.5.2.2. 卷积层

同样，对于卷积层，我们可以在 $\text{\color{red}\colorbox{black}{卷积层之后}}$和 $\text{\color{magenta}\colorbox{black}{非线性激活函数之前}}$ 应用批量规范化。 当卷积有多个输出通道时，我们需要对这些通道的“每个”输出执行批量规范化，每个通道都有自己的拉伸（scale）和偏移（shift）参数，这两个参数都是标量。 假设我们的小批量包含m个样本，并且对于每个通道，卷积的输出具有高度p和宽度q。 那么对于卷积层，我们在每个输出通道的m⋅p⋅q个元素上同时执行每个批量规范化。 因此，在计算平均值和方差时，我们会收集所有空间位置的值，然后在给定通道内应用相同的均值和方差，以便在每个空间位置对值进行规范化。



### 7.5.2.3. 预测过程中的 Batch Normalization

正如我们前面提到的，批量规范化在训练模式和预测模式下的行为通常不同。

- 首先，将训练好的模型用于预测时，我们不再需要样本均值中的噪声以及在微批次上估计每个小批次产生的样本方差了。
- 其次，我们可能没有计算每批标准化统计数据的奢侈。 例如，我们可能需要使用我们的模型对逐个样本进行预测。 一种常用的方法是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。

 可见，和暂退法一样，批量规范化层在训练模式和预测模式下的计算结果也是不一样的。




## 7.5.3. 从零实现

下面，我们从头开始实现一个具有张量的批量规范化层。

```python
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

```

我们现在可以创建一个正确的`BatchNorm`层。 这个层将保持适当的参数：拉伸`gamma`和偏移`beta`,这两个参数将在训练过程中更新。 此外，我们的层将保存均值和方差的移动平均值，以便在模型预测期间随后使用。

撇开算法细节，注意我们实现层的基础设计模式。 通常情况下，我们用一个单独的函数定义其数学原理，比如说`batch_norm`。 然后，我们将此功能集成到一个自定义层中，其代码主要处理数据移动到训练设备（如GPU）、分配和初始化任何必需的变量、跟踪移动平均线（此处为均值和方差）等问题。 为了方便起见，我们并不担心在这里自动推断输入形状，因此我们需要指定整个特征的数量。 不用担心，深度学习框架中的批量规范化API将为我们解决上述问题，我们稍后将展示这一点。


```python
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

```
