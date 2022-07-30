##### 5 Automatic Differentiation

:label:`sec_autograd`

正如我们在 [2.4节](https://zh.d2l.ai/chapter_preliminaries/calculus.html#sec-calculus)中所说的那样，求导是几乎所有深度学习优化算法的关键步骤。 虽然求导的计算很简单，只需要一些基本的微积分。 但对于复杂的模型，手工进行更新是一件很痛苦的事情（而且经常容易出错）。

深度学习框架通过 automatically calculating derivatives，即 *自动微分* （automatic differentiation）来加快求导。 实际中，根据我们设计的模型，系统会构建一个 $\color{red}\text{计算图 (computational graph)}$， 来跟踪计算是哪些数据通过哪些操作组合起来产生输出。 自动微分使系统能够随后反向传播梯度。 这里， *反向传播* （backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

## 5.1 A Simple Example

As a toy example, say that we are interested in (**differentiating the function $y = 2\mathbf{x}^{\top}\mathbf{x}$ with respect to the column vector $\mathbf{x}$.**)
To start, let us create the variable `x` and assign it an initial value.

```python
import torch

x = torch.arange(4.0)
x
```

tensor([0., 1., 2., 3.])
[在我们计算 $y$ 关于 $x$ 的梯度之前，我们$\color{red}需要一个地方来存储梯度$, torch.tensor() $\color{red}数据结构中内部维护$了一个变量（`tensor().grad`，即`x.grad`）来表示这个梯度，可通过 `x.requires_grad_(True)` 开启]
重要的是，我们 $\color{red}不会$ 在每次对一个参数求导时都 $\color{red}分配新的内存$。 因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能很快就会将内存耗尽。 注意，一个标量函数关于向量x的梯度是向量，并且与x具有相同的形状。

```python
x.requires_grad_(True)  # 等价于 Same as `x = torch.arange(4.0, requires_grad=True)`
x.grad  # The default value is None
```

(**Now let us calculate $y$.**)

```python
y = 2 * torch.dot(x, x)
y
```

tensor(28., grad_fn=<MulBackward0>)
Since `x` is a vector of length 4, an dot product of `x` and `x` is performed, yielding the scalar output that we assign to `y`.

Next, [**we can $\color{red}\text{automatically calculate the gradient}$ of `y` 关于每个分量 `x`**] by calling the function for backpropagation `tensor().backward()` and printing the gradient.

```python
y.backward() # 注意，调用的是损失函数 y，而不是 x
x.grad
```

tensor([ 0.,  4.,  8., 12.])
(**The gradient of the function $y = 2\mathbf{x}^{\top}\mathbf{x}$ with respect to $\mathbf{x}$ should be $4\mathbf{x}$.**)
Let us quickly verify that our desired gradient was calculated correctly.

```python
x.grad == 4 * x
```

```
tensor([True, True, True, True])
```

[**Now let us calculate another function of `x`.**] 注意，$\color{red}\text{x.sum()函数的梯度为全1}$

```python
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
# values
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

```
tensor([1., 1., 1., 1.])
```


## 5.2. 非标量变量的反向传播 Backward for Non-Scalar Variables

当`y`不是标量时，向量`y`关于向量`x`的导数的最自然解释是一个矩阵。对于高阶和高维的`y`和`x`，求导的结果可以是一个高阶张量。

然而，虽然这些更奇特的对象确实出现在高级机器学习中（包括深度学习中），但当我们调用 $\color{red}\text{向量}$ 的反向计算时，我们通常会试图计算一批训练样本中 $\color{red}\text{每个组成部分}$ 的损失函数的导数。

Here, (**our $\color{red}\text{intent 目的}$ is**) $\text{\colorbox{black}{\color{yellow}not}}$ to calculate the 微分矩阵 $\text{\colorbox{black}{\color{yellow}but rather}}$ 每个$\color{red}batch$中每个样本单独计算的偏导数之和. 因此，下面我们对 `y.sum()` 进行了反向传播

```python
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
```

tensor([0., 2., 4., 6.])

## 5.3. Detaching Computation 分离计算

有时，we wish $\text{\colorbox{black}{\color{yellow}move}}$ some calculations $\text{\colorbox{black}{\color{yellow}outside}}$ of the recorded computational graph (我们希望将某些计算移动到记录的计算图之外)。 例如，假设`y`是作为`x`的函数计算的，而`z`则是作为`y`和`x`的函数计算的。 想象一下，我们想计算`z`关于`x`的梯度，但由于某种原因，我们 $\color{red}\text{希望将 y 视为一个常数}$， 并且只考虑到`x`在`y`被计算后发挥的作用。

在这里，we can detach `y` to return a $\color{red}new$ variable `u = y.detach`，该变量与`y`具有相同的值， 但丢弃计算图中如何计算`y`的任何信息。 换句话说，梯度不会向后流经`u`到`x`。 因此，下面的反向传播函数计算`z=u*x`关于`x`的偏导数，同时将`u`作为 $\color{red}常数$ 处理， 而不是`z=x*x*x`关于`x`的偏导数。

```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

tensor([True, True, True, True])
由于记录了`y`的计算结果，我们可以随后在`y`上调用反向传播， 得到`y=x*x`关于的`x`的导数，即`2*x`。

```python
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

tensor([True, True, True, True])

## 5.4 Computing the Gradient of Python Control Flow

使用自动微分的一个好处是： 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。 在下面的代码中，`while` 循环的迭代次数和 `if` 语句的结果都取决于输入 `a` 的值。

```python
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Let us compute the gradient.

```python
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

We can now analyze the `f` function defined above.
Note that it is piecewise linear (分段线性) in its input `a`. In other words, for any `a` there exists some constant scalar `k` such that `f(a) = k * a`, where the value of `k` depends on the input `a`. Consequently `d / a` allows us to verify that the gradient is correct.

```python
a.grad == d / a
```

tensor(True)

## Summary

* Deep learning frameworks can automate the calculation of derivatives. To use it, we first attach gradients to those variables with respect to which we desire partial derivatives. We then record the computation of our target value, execute its function for backpropagation, and access the resulting gradient.

## Exercises

1. Why is the second derivative much more expensive to compute than the first derivative?
2. After running the function for backpropagation, immediately run it again and see what happens.
3. In the control flow example where we calculate the derivative of `d` with respect to `a`, what would happen if we changed the variable `a` to a random vector or matrix. At this point, the result of the calculation `f(a)` is no longer a scalar. What happens to the result? How do we analyze this?
4. Redesign an example of finding the gradient of the control flow. Run and analyze the result.
5. Let $f(x) = \sin(x)$. Plot $f(x)$ and $\frac{df(x)}{dx}$, where the latter is computed without exploiting that $f'(x) = \cos(x)$.

[Discussions](https://discuss.d2l.ai/t/35)
