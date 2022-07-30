# 8 RMSProp

:label:`sec_rmsprop`


* RMSProp算法 与 Adagrad算法 非常相似，因为两者都使用梯度的平方来缩放系数。
* RMSProp算法 与 动量法 都使用泄漏平均值 $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$。但是，RMSProp算法使用该技术来调整按系数顺序的预处理器。
* 在实验中，学习率需要由实验者调度。
* 系数γ决定了在调整每坐标比例时历史记录的时长。


One of the key issues in [7_adagrad.md](7_adagrad.md) is that the learning rate decreases at a predefined schedule of effectively $\mathcal{O}(t^{-\frac{1}{2}})$. 虽然这通常适用于凸问题，但对于深度学习中遇到的非凸问题，可能并不理想。 但是，作为一个预处理器，Adagrad算法 $\color{red}按坐标顺序的适应性是非常可取的$。

[[Tieleman &amp; Hinton, 2012]](https://zh.d2l.ai/chapter_references/zreferences.html#tieleman-hinton-2012) 建议以 RMSProp 算法作为将速率调度与坐标自适应学习率分离的简单修复方法。 $\text{\color{yellow}\colorbox{black}{The issue}}$ is that Adagrad accumulates the squares of the gradient $\mathbf{g}_t$ into a state vector $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$. As a result $\mathbf{s}_t$ keeps on growing without bound due to the lack of normalization, essentially linearly as the algorithm converges.

- $\text{\color{yellow}\colorbox{black}{One way}}$ of fixing this problem would be to use $\mathbf{s}_t / t$. For reasonable distributions of $\mathbf{g}_t$ this will converge. $\color{magenta}Unfortunately$ 遗憾的是，限制行为生效可能需要很长时间，因为该流程记住了值的完整轨迹。
- $\text{\color{yellow}\colorbox{black}{An alternative}}$ is to $\color{red}按动量法中的方式使用泄漏平均值$, i.e., $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ for some parameter $\gamma > 0$. Keeping all other parts unchanged yields RMSProp.

## 8.1 The Algorithm

让我们详细写出这些方程式。

$$
\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}

$$

The constant $\epsilon > 0$ is typically set to $10^{-6}$ to ensure that we do not suffer from division by zero or overly large step sizes. 鉴于这种扩展，我们现在可以自由控制学习率η，而不考虑基于每个坐标应用的缩放。 就泄漏平均值而言，我们可以采用与之前在动量法中适用的相同推理。 Expanding the definition of $\mathbf{s}_t$ yields

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}

$$

As before in [6_momentum.md](6_momentum.md) we use $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$. Hence the sum of weights is normalized to $1$ with a half-life time of an observation of $\gamma^{-1}$. Let us visualize the weights for the past 40 time steps for various choices of $\gamma$.

```python
import math
import torch
from d2l import torch as d2l
```

```python
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = torch.arange(40).detach().numpy()
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

![../_images/output_rmsprop_251805_6_0.svg](https://zh.d2l.ai/_images/output_rmsprop_251805_6_0.svg)

## 8.2 Implementation from Scratch

As before we use the quadratic function $f(\mathbf{x})=0.1x_1^2+2x_2^2$ to observe the trajectory of RMSProp. Recall that in [7_adagrad.md](7_adagrad.md), when we used Adagrad with a learning rate of 0.4, the variables moved only very slowly in the later stages of the algorithm since the learning rate decreased too quickly. Since $\eta$ is controlled separately this does not happen with RMSProp.

```python
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

epoch 20, x1: -0.010599, x2: 0.000000
/home/d2l-worker/miniconda3/envs/d2l-en-release-0/lib/python3.8/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
![../_images/output_rmsprop_251805_18_1.svg](https://zh.d2l.ai/_images/output_rmsprop_251805_18_1.svg)

Next, we implement RMSProp to be used in a deep network. This is equally straightforward.

```python
def init_rmsprop_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)
```
```python
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```
We set the initial learning rate to 0.01 and the weighting term $\gamma$ to 0.9. That is, $\mathbf{s}$ aggregates on average over the past $1/(1-\gamma) = 10$ observations of the square gradient.

```python
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```
loss: 0.246, 0.015 sec/epoch
![../_images/output_rmsprop_251805_42_1.svg](https://zh.d2l.ai/_images/output_rmsprop_251805_42_1.svg)

## Concise Implementation

Since RMSProp is a rather popular algorithm it is also available in the `Trainer` instance. All we need to do is instantiate it using an algorithm named `rmsprop`, assigning $\gamma$ to the parameter `gamma1`.

```python
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```
loss: 0.243, 0.013 sec/epoch
![../_images/output_rmsprop_251805_54_1.svg](https://zh.d2l.ai/_images/output_rmsprop_251805_54_1.svg)

## Summary

* RMSProp is very similar to Adagrad insofar as both use the square of the gradient to scale coefficients.
* RMSProp shares with momentum the leaky averaging. However, RMSProp uses the technique to adjust the coefficient-wise preconditioner.
* The learning rate needs to be scheduled by the experimenter in practice.
* The coefficient $\gamma$ determines how long the history is when adjusting the per-coordinate scale.

## Exercises

1. What happens experimentally if we set $\gamma = 1$? Why?
2. Rotate the optimization problem to minimize $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. What happens to the convergence?
3. Try out what happens to RMSProp on a real machine learning problem, such as training on Fashion-MNIST. Experiment with different choices for adjusting the learning rate.
4. Would you want to adjust $\gamma$ as optimization progresses? How sensitive is RMSProp to this?

[Discussions](https://discuss.d2l.ai/t/1074)
