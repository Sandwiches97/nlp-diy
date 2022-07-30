# 7 Adagrad

:label:`sec_adagrad`


* AdaGrad算法会在 $\color{magenta}单个坐标层面$ 动态降低学习率。
* AdaGrad算法利用梯度的大小作为调整进度速率的手段：用较小的学习率来补偿带有较大梯度的坐标。
* 在深度学习问题中，由于内存和计算限制，计算准确的二阶导数通常是不可行的。梯度可以作为一个有效的代理。
* 如果优化问题的结构相当不均匀，AdaGrad算法可以帮助缓解扭曲。
* AdaGrad算法 $\color{magenta}对于稀疏特征特别有效$，因为分母对应每个参数有个累加的过程，经常更新的参数，分母较大。
* 在深度学习问题上，AdaGrad算法有时在降低学习率方面可能过于剧烈。我们将在 [11.10节](https://zh.d2l.ai/chapter_optimization/adam.html#sec-adam)一节讨论缓解这种情况的策略。


Let us begin by considering learning problems with features that occur infrequently.

## 7.1 Sparse Features and Learning Rates

Imagine that we are training a language model (语言模型). To get good accuracy we typically want to decrease the learning rate as we keep on training, usually at a rate of $\mathcal{O}(t^{-\frac{1}{2}})$ or slower. 现在讨论关于 $\text{\color{red}\colorbox{black}{稀疏特征}}$（即只在偶尔出现的特征）的模型训练，这对 $\text{\color{red}\colorbox{black}{NLP}}$ 来说很常见。 例如，我们看到 “preconditioning” 这个词比“learning”这个词的可能性要小得多。 但是，它在计算广告学和个性化协同过滤等其他领域也很常见。

只有在这些 $\text{\color{black}\colorbox{yellow}{不常见的特征出现时，与其相关的参数才会得到有意义的更新}}$。 鉴于学习率下降，我们可能最终会面临这样的情况：常见特征的参数相当迅速地收敛到最佳值，而对于不常见的特征，我们仍缺乏足够的观测以确定其最佳值。 换句话说，

- 学习率要么对于 $\text{\color{black}\colorbox{yellow}{常见特征}}$ 而言降低太慢，
- 要么对于 $\text{\color{black}\colorbox{yellow}{不常见特征}}$ 而言降低太快。

A possible hack to redress this issue would be to $\text{\color{magenta}{count the number of times}}$ we see a particular feature and to use this as a clock $\text{\color{magenta}for adjusting learning rates}$. That is, rather than choosing a learning rate of the form $\eta = \frac{\eta_0}{\sqrt{t + c}}$ we could use $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$. Here $s(i, t)$ counts the number of nonzeros for feature $i$ that we have observed up to time $t$. 这其实很容易实施且不产生额外损耗。$\text{\color{yellow}\colorbox{black}{However, it fails}}$ whenever we do not quite have sparsity but rather just data where the gradients are often very small and only rarely large. 毕竟，目前尚不清楚在哪里可以划定是否符合观察到的特征。

Adagrad by [[Duchi et al., 2011]](https://zh.d2l.ai/chapter_references/zreferences.html#duchi-hazan-singer-2011) addresses this by replacing the rather crude counter $s(i, t)$ by an aggregate of the squares of previously observed gradients. In particular, it uses $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$ as a means $\text{\color{yellow}\colorbox{black}{to adjust the learning rate}}$. 这有两个好处：

- 首先，我们不再需要决定梯度何时算足够大。
- 其次，它会随梯度的大小自动变化。通常对应于较大梯度的坐标会显著缩小，而其他梯度较小的坐标则会得到更平滑的处理。

在实际应用中，它促成了计算广告学及其相关问题中非常有效的优化程序。 但是，它遮盖了AdaGrad 固有的一些额外优势 that are best understood in the context of preconditioning.

## 7.2 Preconditioning

凸优化问题有助于分析算法的特点。 毕竟对于大多数非凸问题来说，获得有意义的理论保证很难，但是直觉和洞察往往会延续。 Let us look at the problem of minimizing $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$.

As we saw in [6_momentum.md](6_momentum.md), it is possible to rewrite this problem in terms of its eigendecomposition $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ to arrive at a much simplified problem where each coordinate (坐标) can be solved individually:

$$
f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b. \text{  相似对角化}

$$

Here we used $\mathbf{x} = \mathbf{U} \mathbf{x}$ and consequently $\mathbf{c} = \mathbf{U} \mathbf{c}$. The modified problem has as its minimizer $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$ and minimum value $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$. This is much easier to compute since $\boldsymbol{\Lambda}$ is a diagonal matrix containing the eigenvalues of $\mathbf{Q}$.

If we perturb $\color{magenta}\mathbf{c}$ $\color{Red}\text{slightly}$ $\text{\color{yellow}\colorbox{black}{we would hope}}$ to find only $\color{Red}\text{slight changes}$ in the minimizer of $\color{magenta}f$. 遗憾的是，情况并非如此。 While slight changes in $\mathbf{c}$ lead to equally slight changes in $\bar{\mathbf{c}}$, this is not the case for the minimizer of $f$ (and of $\bar{f}$ respectively).

- Whenever the eigenvalues $\boldsymbol{\Lambda}_i$ are $\color{Red}\text{large}$ we will see only small changes in $\bar{x}_i$ and in the minimum of $\bar{f}$.
- Conversely, for $\color{Red}\text{small}$ $\boldsymbol{\Lambda}_i$ changes in $\bar{x}_i$ can be $\text{\color{yellow}\colorbox{black}{dramatic}}$.

The ratio between the largest and the smallest eigenvalue is called the $\text{\color{red}\colorbox{black}{condition number (*条件数* )}}$ of an optimization problem.

$$
\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.

$$

If the condition number $\kappa$ is large, it is difficult to solve the optimization problem accurately. We need to ensure that we are careful in getting a large dynamic range of values right. Our analysis leads to an obvious, albeit somewhat naive question: couldn't we simply "fix" the problem by distorting the space such that all eigenvalues are $1$. In theory this is quite easy: we only need the eigenvalues and eigenvectors of $\mathbf{Q}$ to rescale the problem from $\mathbf{x}$ to one in $\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$. In the new coordinate system $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ could be simplified to $\|\mathbf{z}\|^2$. 可惜，这是一个相当不切实际的想法。 一般而言，计算特征值和特征向量要比解决实际问题“贵”得多。

虽然准确计算特征值可能会很昂贵，但即便只是大致猜测并计算它们，也可能已经比不做任何事情好得多。 特别是，我们可以使用 the diagonal entries of $Q$ 并相应地重新缩放它。 这比计算特征值开销小的多。

$$
\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).

$$

In this case we have $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ and specifically $\tilde{\mathbf{Q}}_{ii} = 1$ for all $i$. 在大多数情况下，这大大简化了条件数。 例如我们之前讨论的案例，它将完全消除眼下的问题，因为问题是轴对齐的。

遗憾的是，我们还面临另一个问题：在深度学习中，我们通常情况甚至无法计算目标函数的二阶导数: for $\mathbf{x} \in \mathbb{R}^d$ the second derivative even on a minibatch may require $\mathcal{O}(d^2)$ space and work to compute, thus making it practically infeasible. AdaGrad算法巧妙的思路是，使用一个代理来表示黑塞矩阵（Hessian）的对角线，既相对易于计算又高效。

In order to see why this works, let us look at $\bar{f}(\bar{\mathbf{x}})$. We have that

$$
\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),

$$

where $\bar{\mathbf{x}}_0$ is the minimizer of $\bar{f}$. Hence the magnitude of the gradient depends both on $\boldsymbol{\Lambda}$ and the distance from optimality. If $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ didn't change, this would be all that's needed. After all, in this case the magnitude of the gradient $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ suffices. 由于AdaGrad算法是一种随机梯度下降算法，所以即使是在最佳值中，我们也会看到具有非零方差的梯度。 因此，我们可以放心地使用梯度的方差作为黑塞矩阵比例的廉价替代。 详尽的分析（要花几页解释）超出了本节的范围，请读者参考 [[Duchi et al., 2011]](https://zh.d2l.ai/chapter_references/zreferences.html#duchi-hazan-singer-2011)。

## 7.3 The Algorithm

Let us formalize the discussion from above. We use the variable $\mathbf{s}_t$ to accumulate past gradient variance as follows.

$$
\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}

$$

Here the operation are applied $\text{\color{black}\colorbox{yellow}{coordinate wise}}$. That is, $\mathbf{v}^2$ has entries $v_i^2$. Likewise $\frac{1}{\sqrt{v}}$ has entries $\frac{1}{\sqrt{v_i}}$ and $\mathbf{u} \cdot \mathbf{v}$ has entries $u_i v_i$. As before $\eta$ is the learning rate and $\epsilon$ is an additive constant that ensures that we do not divide by $0$. Last, we initialize $\mathbf{s}_0 = \mathbf{0}$.

Just like in the case of momentum we need to keep track of an auxiliary variable, in this case to allow for an individual learning rate per coordinate (允许每个坐标有单独的学习率). This does not increase the cost of Adagrad significantly relative to SGD, simply since the main cost is typically to compute $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ and its derivative.

Note that accumulating squared gradients in $\mathbf{s}_t$ means that $\mathbf{s}_t$ grows essentially at linear rate (somewhat slower than linearly in practice, since the gradients initially diminish). This leads to an $\mathcal{O}(t^{-\frac{1}{2}})$ learning rate, albeit adjusted on a per coordinate basis. 对于凸问题，这完全足够了。 然而，在深度学习中，我们可能希望更慢地降低学习率。 这引出了许多AdaGrad算法的变体，我们将在后续章节中讨论它们。 眼下让我们先看看它在二次凸问题中的表现如何。 我们仍然以同一函数为例：

$$
f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.

$$

我们将使用与之前相同的学习率来实现AdaGrad算法，即 $η=0.4$。 可以看到，自变量的迭代轨迹较平滑。 但由于 $s_t$ 的累加效果使学习率不断衰减，自变量在迭代后期的移动幅度较小。

```python
%matplotlib inline
import math
import torch
from d2l import torch as d2l
```

```python
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

epoch 20, x1: -2.382563, x2: -0.158591


![../_images/output_adagrad_2fb0ed_3_1.svg](https://zh.d2l.ai/_images/output_adagrad_2fb0ed_3_1.svg)

我们将学习率提高到2，可以看到更好的表现。 这已经表明，即使在无噪声的情况下，学习率的降低可能相当剧烈，我们需要确保参数能够适当地收敛。

```python
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

epoch 20, x1: -0.002295, x2: -0.000000

![../_images/output_adagrad_2fb0ed_15_1.svg](https://zh.d2l.ai/_images/output_adagrad_2fb0ed_15_1.svg)

## 7.4 Implementation from Scratch

同 momentum method 一样，AdaGrad算法需要对每个自变量维护同它一样形状的状态变量。

```python
def init_adagrad_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

与 [11.5节](https://zh.d2l.ai/chapter_optimization/minibatch-sgd.html#sec-minibatch-sgd)一节中的实验相比，这里使用更大的学习率来训练模型。

```python
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

loss: 0.242, 0.016 sec/epoch
![../_images/output_adagrad_2fb0ed_42_1.svg](https://zh.d2l.ai/_images/output_adagrad_2fb0ed_42_1.svg)

## 7.5 Concise Implementation

Using the `Trainer` instance of the algorithm `adagrad`, we can invoke the Adagrad algorithm in Gluon.

```python
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

loss: 0.242, 0.016 sec/epoch
![../_images/output_adagrad_2fb0ed_54_1.svg](https://zh.d2l.ai/_images/output_adagrad_2fb0ed_54_1.svg)

## Summary

* Adagrad decreases the learning rate dynamically on a per-coordinate basis.
* It uses the magnitude of the gradient as a means of adjusting how quickly progress is achieved - coordinates with large gradients are compensated with a smaller learning rate.
* Computing the exact second derivative is typically infeasible in deep learning problems due to memory and computational constraints. The gradient can be a useful proxy.
* If the optimization problem has a rather uneven structure Adagrad can help mitigate the distortion.
* Adagrad is particularly effective for sparse features where the learning rate needs to decrease more slowly for infrequently occurring terms.
* On deep learning problems Adagrad can sometimes be too aggressive in reducing learning rates. We will discuss strategies for mitigating this in the context of :numref:`sec_adam`.

## Exercises

1. Prove that for an orthogonal matrix $\mathbf{U}$ and a vector $\mathbf{c}$ the following holds: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. Why does this mean that the magnitude of perturbations does not change after an orthogonal change of variables?
2. Try out Adagrad for $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ and also for the objective function was rotated by 45 degrees, i.e., $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Does it behave differently?
3. Prove [Gerschgorin's circle theorem](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) which states that eigenvalues $\lambda_i$ of a matrix $\mathbf{M}$ satisfy $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$ for at least one choice of $j$.
4. What does Gerschgorin's theorem tell us about the eigenvalues of the diagonally preconditioned matrix $\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$?
5. Try out Adagrad for a proper deep network, such as :numref:`sec_lenet` when applied to Fashion MNIST.
6. How would you need to modify Adagrad to achieve a less aggressive decay in learning rate?

[Discussions](https://discuss.d2l.ai/t/1072)
