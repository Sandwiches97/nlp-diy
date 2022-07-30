# 4 Softmax Regression

:label:`sec_softmax`

* softmax运算 获取一个向量并将其映射为概率。
* softmax回归 适用于分类问题，它使用了softmax运算中输出类别的概率分布。
* 交叉熵是一个衡量两个概率分布之间差异的很好的度量，它测量给定模型编码数据所需的比特数。

在 [3.1节](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#sec-linear-regression)中我们介绍了线性回归。 随后，在 [3.2节](https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html#sec-linear-scratch)中我们从头实现线性回归。 然后，在 [3.3节](https://zh.d2l.ai/chapter_linear-networks/linear-regression-concise.html#sec-linear-concise)中我们使用深度学习框架的高级API简洁实现线性回归。

$\color{magenta}\text{Regression}$ 可以用于预测 *$\color{magenta}\text{多少}$* 的问题。 比如预测房屋被售出价格，或者棒球队可能获得的胜场数，又或者患者住院的天数。

事实上，我们也对 $\color{red}\text{Classification}$ 问题感兴趣：不是问“多少”，而是问 “$\color{red}\text{哪一个}$”：

* 某个电子邮件是否属于垃圾邮件文件夹？
* 某个用户可能*注册*或*不注册*订阅服务？
* 某个图像描绘的是驴、狗、猫、还是鸡？
* 某人接下来最有可能看哪部电影？

通常，机器学习实践者用 $\color{red}\text{Classification}$ 这个词来描述两个有微妙差别的问题：

1. 我们只对样本的“ $\color{red}\text{硬性}$”类别感兴趣，即属于哪个类别；
2. 我们希望得到“ $\color{magenta}\text{软性}$ ”类别，即得到属于每个类别的 $\color{magenta}\text{概率}$ 。

这两者的界限往往很模糊。其中的一个原因是：即使我们只关心硬类别，我们仍然使用软类别的模型。

## 4.1 Classification Problem

:label:`subsec_classification-problem`

我们从一个图像分类问题开始。 假设每次输入是一个 `2×2` 的灰度图像。 我们可以用一个标量表示每个像素值，每个图像对应四个特征 $x_1,x_2,x_3,x_4$。 此外，假设每个图像属于类别 `“猫”，“鸡”, “狗”` 中的一个。

接下来，我们要选择如何表示标签。 我们有两个明显的选择：

- 最直接的想法是选择 $\color{red}y \in \{1, 2, 3\}$, where the integers represent $\{\text{dog}, \text{cat}, \text{chicken}\}$ respectively.
  这是在计算机上存储此类信息的有效方法。
- 如果类别间有一些 $\color{magenta}\text{自然顺序}$， 比如说我们试图预测 $\{\text{baby}$, $\text{toddler}$, $\text{adolescent}$, $\text{young adult}$, $\text{adult}$, $\text{geriatric}\}$, 那么将这个问题转变为 $\color{magenta}\text{回归问题}$，并且保留这种格式是有意义的。

但是一般的 $\color{red}\text{分类问题}$ 并不与类别之间的自然顺序有关。 幸运的是，统计学家很早以前就发明了一种表示分类数据的简单方法： $\color{red}\text{独热编码 （one-hot encoding）}$。 独热编码是一个向量，它的分量和类别一样多。 类别对应的分量设置为1，其他所有分量设置为0。 在我们的例子中，标签y将是一个三维向量， 其中(1,0,0)对应于“猫”、(0,1,0)对应于“鸡”、(0,0,1)对应于“狗”：

$$
y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.

$$

## 4.2 Network Architecture

为了估计所有可能类别的条件概率，我们需要一个有多个输出的模型，每个类别对应一个输出。 为了解决线性模型的分类问题，我们需要和输出一样多的 *仿射函数* （affine function）。 每个输出对应于它自己的仿射函数。 在我们的例子中，由于我们有 $\color{yellow}\text{\colorbox{black}{4 个特征}}$ 和 $\color{yellow}\text{\colorbox{black}{3 个可能的输出类别}}$， 我们将需要 $12$ 个标量来表示权重（带下标的 $w$ ）， $3$ 个标量来表示偏置（带下标的 $b$）。 下面我们为每个输入计算三个 *未规范化的预测* （logit）： $o_1, o_2$, and $o_3$。

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}

$$

我们可以用神经网络图 [图3.4.1](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html#fig-softmaxreg)来描述这个计算过程。 与线性回归一样，softmax回归也是一个单层神经网络。
And since the calculation of each output, $o_1, o_2$, and $o_3$, depends on all inputs, $x_1$, $x_2$, $x_3$, and $x_4$, the output layer of $\color{red}\text{softmax regression}$ can also $\color{yellow}\text{\colorbox{black}{be described as}}$ $\color{red}\text{fully-connected layer}$.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://zh.d2l.ai/_images/softmaxreg.svg" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Softmax regression is a single-layer neural network.
  	</div>
</center>

To express the model more compactly, we can use linear algebra notation 线性代数符号.
In vector form, we arrive at $\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$, a form better suited both for mathematics, and for writing code. Note that we have gathered all of our weights into a $3 \times 4$ matrix and that for features of a given data example $\mathbf{x}$, our outputs are given by a matrix-vector product of our weights by our input features
plus our biases $\mathbf{b}$.

## 4.3 Parameterization Cost 参数代价 of FC Layers

:label:`subsec_parameterization-cost-fc-layers`

正如我们将在后续章节中看到的，在深度学习中，全连接层无处不在。 然而，顾名思义，全连接层是“完全”连接的，可能有很多可学习的参数。
Specifically, for any $\color{green}\text{\colorbox{white}{fully-connected layer}}$ with $d$ inputs and $q$ outputs, the parameterization cost is $\mathcal{O}(dq)$, which can be prohibitively high in practice. $\color{red}Fortunately$, this cost of transforming $d$ inputs into $q$ outputs can be reduced to $\color{red}\mathcal{O}(\frac{dq}{n})$, where the hyperparameter $n$ can be flexibly specified by us to balance between parameter saving and model effectiveness in real-world applications [[Zhang et al., 2021]](https://zh.d2l.ai/chapter_references/zreferences.html#zhang-tay-zhang-ea-2021)。

## 4.4 Softmax Operation

:label:`subsec_softmax_operation`

现在我们将优化参数以最大化观测数据的概率。 为了得到预测结果，我们将设置一个阈值，如选择具有最大概率的标签。
We will optimize our parameters to produce probabilities that maximize the likelihood of the observed data. Then, to generate predictions, we will set a threshold, for example, choosing the label with the maximum predicted probabilities.

Put formally, we would like any output $\hat{y}_j$ to be interpreted as the probability that a given item belongs to class $j$. Then we can choose the class with the largest output value as our prediction $\operatorname*{argmax}_j y_j$.
For example, if $\hat{y}_1$, $\hat{y}_2$, and $\hat{y}_3$ are 0.1, 0.8, and 0.1, respectively, then we predict category 2, which (in our example) represents "chicken".

然而我们能否将未规范化的预测o直接视作我们感兴趣的输出呢？ 答案是否定的。 因为将线性层的输出直接视为概率时存在一些问题：

- 一方面，我们没有限制这些输出数字的总和为1。
- 另一方面，根据输入的不同，它们可以为负值。 这些违反了 [2.6节](https://zh.d2l.ai/chapter_preliminaries/probability.html#sec-prob)中所说的概率基本公理。

要将输出视为概率，我们必须保证在任何数据上的输出都是非负的且总和为1。 此外，我们需要一个训练目标，来鼓励模型精准地估计概率。 在分类器输出0.5的所有样本中，我们希望这些样本有一半实际上属于预测的类。 这个属性叫做 *校准* （calibration）。

社会科学家邓肯·卢斯于1959年在 *选择模型* （choice model）的理论基础上 发明的*softmax函数*正是这样做的： softmax函数将未规范化的预测变换为非负并且总和为1，同时要求模型保持可导。 我们首先对每个未规范化的预测求幂，这样可以确保输出非负。 为了确保最终输出的总和为1，我们再对每个求幂后的结果除以它们的总和。如下式：

$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}.

$$

It is easy to see $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$ with $0 \leq \hat{y}_j \leq 1$ for all $j$. Thus, $\hat{\mathbf{y}}$ is a proper probability distribution whose element values can be interpreted accordingly. softmax 运算不会改变未规范化的预测 $o$ 之间的顺序，只会确定分配给每个类别的概率。 因此，在预测过程中，我们仍然可以用下式来选择最有可能的类别。

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.

$$

尽管$\color{red}softmax是一个非线性函数$，但 $\color{red}\text{\colorbox{white}{softmax 回归}}$ 的输出仍然由输入特征的仿射变换决定。 因此，$\color{red}\text{\colorbox{white}{softmax回归}}$ 是一个 $\color{red}\text{\colorbox{white}{线性模型 （linear model）}}$。

## 4.5 Vectorization for Minibatches

:label:`subsec_softmax_vectorization`

矢量化：实际上是用到了SIMD （**以同步方式，在同一时间内执行同一条指令**），做到了并行运算。这实际上就是矢量运算。

To improve computational efficiency and take advantage of GPUs, we typically carry out vector calculations (矢量计算) for $\color{red}\text{minibatches of data}$. Assume that we are given a minibatch $\mathbf{X}$ of examples with feature dimensionality (number of inputs) $d$ and batch size $n$. Moreover, assume that we have $q$ categories in the output. Then the minibatch features $\mathbf{X}$ are in $\mathbb{R}^{n \times d}$, weights $\mathbf{W} \in \mathbb{R}^{d \times q}$, and the bias satisfies $\mathbf{b} \in \mathbb{R}^{1\times q}$.

$$
\begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned}

$$

This accelerates the dominant operation into a $\color{red}\text{matrix-matrix product}$ $\mathbf{X} \mathbf{W}$ $\color{yellow}\text{\colorbox{black}{vs.}}$ the $\color{red}\text{matrix-vector}$ products we would be executing if we processed one example at a time. Since each row in $\mathbf{X}$ represents a data example, the softmax operation itself can be computed *rowwise*: for each row of $\mathbf{O}$, exponentiate all entries and then normalize them by the sum. Triggering broadcasting during the summation $\mathbf{X} \mathbf{W} + \mathbf{b}$ in [(3.4.5)](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html#equation-eq-minibatch-softmax-reg), both the minibatch logits $\mathbf{O}$ and output probabilities $\hat{\mathbf{Y}}$ are $n \times q$ matrices.

## 4.6 Loss Function

接下来，我们需要一个损失函数来度量预测的效果。 我们将使用最大似然估计，这与在线性回归 （ [3.1.3节](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#subsec-normal-distribution-and-squared-loss)） 中的方法相同。

### 4.6.1 Log-Likelihood

The softmax function gives us a vector $\hat{\mathbf{y}}$, which we can interpret as estimated conditional probabilities
of each class given any input $\mathbf{x}$, e.g., $\hat{y}_1$ = $P(y=\text{cat} \mid \mathbf{x})$.
Suppose that the $\color{yellow}\text{\colorbox{black}{entire dataset}}$ $\{\mathbf{X}, \mathbf{Y}\}$ has $n$ examples, where the example indexed by $i$ consists of a feature vector $\mathbf{x}^{(i)}$ and a one-hot label vector $\mathbf{y}^{(i)}$. We can compare the estimates with reality by checking how probable the actual classes are according to our model, given the features:

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).

$$

According to maximum likelihood estimation, we maximize $P(\mathbf{Y} \mid \mathbf{X})$, which is equivalent to minimizing the negative log-likelihood:

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = {\color{red}\sum_{i=1}^n} {\color{magenta}-\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})}
= {\color{red}\sum_{i=1}^n} {\color{magenta}l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)})},

$$

where for any pair of label $\mathbf{y}$ and model prediction $\hat{\mathbf{y}}$ over $\color{magenta}\text{\colorbox{black}{q classes}}$, the loss function $l$ is

$$
\color{magenta}l(\mathbf{y}, \hat{\mathbf{y}}) = H_{p^{predict}}(p^{true}) = - {\color{green}\sum_{j=1}^q y_j} \log \hat{y}_j.\tag{3.4.8}

$$

For reasons explained later on, the loss function in [Eq 3.4.8]() is commonly called the *$\color{magenta}\text{\colorbox{black}{cross-entropy loss (非对称)}}$*. Since $\mathbf{y}$ is a $\color{magenta}\text{\colorbox{black}{one-hot vector,}}$ of length $q$, the sum over all its coordinates $j$ vanishes for all but one term $\color{green}（\sum_{j=1}^q y_j \text{相当于0-1编码，选择作用）}$ . Since all $\hat{y}_j$ are predicted probabilities, 它们的对数永远不会大于0。 因此，如果正确地预测实际标签，即如果实际标签 $P(\mathbf{y} \mid \mathbf{x}) = 1$， 则损失函数不能进一步最小化。注意，这往往是不可能的。 例如，数据集中可能存在标签噪声（比如某些样本可能被误标）， 或输入特征没有足够的信息来完美地对每一个样本分类。

### 4.6.2 Softmax and Derivatives 导数

:label:`subsec_softmax_and_derivatives`

由于softmax和相关的损失函数很常见， 因此我们需要更好地理解它的计算方式。 将 [(3.4.3)](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html#equation-eq-softmax-y-and-o)代入损失 [(3.4.8)](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html#equation-eq-l-cross-entropy)中。 利用softmax的定义，我们得到：

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= {\color{red}\sum_{j=1}^q y_j} \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}

$$

红色公式是独热编码。To understand a bit better what is going on, consider the derivative with respect to any logit $o_j$. We get

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.

$$

换句话说，导数是我们 softmax模型 分配的概率与实际发生的情况（由独热标签向量表示）之间的差异。 从这个意义上讲，这与我们在回归中看到的非常相似， 其中梯度是观测值y和估计值y^之间的差异。 这不是巧合，在任何指数族分布模型中 （参见[本书附录中关于数学分布的一节](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/distributions.html)）， 对数似然的梯度正是由此得出的。 这使梯度计算在实践中变得容易很多。

### 4.6.3 Cross-Entropy Loss

现在让我们考虑整个结果分布的情况，即观察到的不仅仅是一个结果。 对于标签y，我们可以使用与以前相同的表示形式。 唯一的区别是，我们现在用一个概率向量表示，如 $(0.1,0.2,0.7)$， 而不是仅包含二元项的向量 $(0,0,1)$。 我们使用 [(3.4.8)](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html#equation-eq-l-cross-entropy)来定义损失 $l$， 它是所有标签分布的预期损失值。 此损失称为 *交叉熵损失* （cross-entropy loss），它是分类问题最常用的损失之一。 本节我们将通过介绍信息论基础来理解交叉熵损失。 如果你想了解更多信息论的细节，你可以进一步参考 [本书附录中关于信息论的一节](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html)。

## 4.7 Information Theory Basics

:label:`subsec_info_theory_basics`

*Information theory* deals with the problem of encoding, decoding, transmitting, and manipulating information (also known as data) in as concise form as possible.

### Entropy

The central idea in information theory is to quantify the information content in data.
This quantity places a hard limit on our ability to compress the data.
In information theory, this quantity is called the *entropy* of a distribution $P$,
and it is captured by the following equation:

$$
H[P] = \sum_j - P(j) \log P(j).

$$

:eqlabel:`eq_softmax_reg_entropy`

One of the fundamental theorems of information theory states
that in order to encode data drawn randomly from the distribution $P$,
we need at least $H[P]$ "nats" to encode it.
If you wonder what a "nat" is, it is the equivalent of bit
but when using a code with base $e$ rather than one with base 2.
Thus, one nat is $\frac{1}{\log(2)} \approx 1.44$ bit.

### Surprisal

You might be wondering what compression has to do with prediction.
Imagine that we have a stream of data that we want to compress.
If it is always easy for us to predict the next token,
then this data is easy to compress!
Take the extreme example where every token in the stream always takes the same value.
That is a very boring data stream!
And not only it is boring, but it is also easy to predict.
Because they are always the same, we do not have to transmit any information
to communicate the contents of the stream.
Easy to predict, easy to compress.

However if we cannot perfectly predict every event,
then we might sometimes be surprised.
Our surprise is greater when we assigned an event lower probability.
Claude Shannon settled on $\log \frac{1}{P(j)} = -\log P(j)$
to quantify one's *surprisal* at observing an event $j$
having assigned it a (subjective) probability $P(j)$.
The entropy defined in :eqref:`eq_softmax_reg_entropy` is then the *expected surprisal*
when one assigned the correct probabilities
that truly match the data-generating process.

### Cross-Entropy Revisited

So if entropy is level of surprise experienced
by someone who knows the true probability,
then you might be wondering, what is cross-entropy?
The cross-entropy *from* $P$ *to* $Q$, denoted $H(P, Q)$,
is the expected surprisal of an observer with subjective probabilities $Q$
upon seeing data that were actually generated according to probabilities $P$.
The lowest possible cross-entropy is achieved when $P=Q$.
In this case, the cross-entropy from $P$ to $Q$ is $H(P, P)= H(P)$.

In short, we can think of the cross-entropy classification objective
in two ways: (i) as maximizing the likelihood of the observed data;
and (ii) as minimizing our surprisal (and thus the number of bits)
required to communicate the labels.

## Model Prediction and Evaluation

After training the softmax regression model, given any example features,
we can predict the probability of each output class.
Normally, we use the class with the highest predicted probability as the output class.
The prediction is correct if it is consistent with the actual class (label).
In the next part of the experiment,
we will use *accuracy* to evaluate the model's performance.
This is equal to the ratio between the number of correct predictions and the total number of predictions.

## Summary

* The softmax operation takes a vector and maps it into probabilities.
* Softmax regression applies to classification problems. It uses the probability distribution of the output class in the softmax operation.
* Cross-entropy is a good measure of the difference between two probability distributions. It measures the number of bits needed to encode the data given our model.

## Exercises

1. We can explore the connection between exponential families and the softmax in some more depth.
   1. Compute the second derivative of the cross-entropy loss $l(\mathbf{y},\hat{\mathbf{y}})$ for the softmax.
   2. Compute the variance of the distribution given by $\mathrm{softmax}(\mathbf{o})$ and show that it matches the second derivative computed above.
2. Assume that we have three classes which occur with equal probability, i.e., the probability vector is $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
   1. What is the problem if we try to design a binary code for it?
   2. Can you design a better code? Hint: what happens if we try to encode two independent observations? What if we encode $n$ observations jointly?
3. Softmax is a misnomer for the mapping introduced above (but everyone in deep learning uses it). The real softmax is defined as $\mathrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
   1. Prove that $\mathrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$.
   2. Prove that this holds for $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$, provided that $\lambda > 0$.
   3. Show that for $\lambda \to \infty$ we have $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$.
   4. What does the soft-min look like?
   5. Extend this to more than two numbers.

[Discussions](https://discuss.d2l.ai/t/46)
