## 1. 数据操作 data manipulation

为了能够完成各种数据操作，我们需要某种方法来存储和操作数据。 通常，我们需要做两件重要的事：

- （1）获取数据；
- （2）将数据读入计算机后对其进行处理。 如果没有某种方法来存储数据，那么获取数据是没有意义的。

首先，我们介绍n维数组，也称为 *张量* （tensor）。 使用过Python中NumPy计算包的读者会对本部分很熟悉。 无论使用哪个深度学习框架，它的 *张量类* （在MXNet中为`ndarray`， 在 PyTorch 和 TensorFlow 中为`Tensor`）都与Numpy的`ndarray`类似。 但深度学习框架又比Numpy的`ndarray`多一些重要功能：

- 首先，GPU很好地支持加速计算，而NumPy仅支持CPU计算；
- 其次，张量类支持自动微分。

这些功能使得张量类更适合深度学习。 如果没有特殊说明，本书中所说的张量均指的是张量类的实例。

### torch.cat()

我们也可以把多个张量 *连结* （concatenate）在一起， 把它们端对端地叠起来形成一个更大的张量。 我们只需要提供张量列表，并给出沿哪个轴连结。

下面的例子分别演示了当我们沿行（轴-0，形状的第一个元素） 和按列（轴-1，形状的第二个元素）连结两个矩阵时，会发生什么情况。

```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

我们可以看到，第一个输出张量的轴-0长度（6）是两个输入张量轴-0长度的总和（3+3）； 第二个输出张量的轴-1长度（8）是两个输入张量轴-1长度的总和（4+4）。

````
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [ 2.,  1.,  4.,  3.],
         [ 1.,  2.,  3.,  4.],
         [ 4.,  3.,  2.,  1.]]),
 tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))

````

### 通过*逻辑运算符*构建二元张量

以 `X == Y` 为例： 对于每个位置，如果 `X` 和 `Y` 在该位置相等，则新张量中相应项的值为1。 这意味着逻辑语句`X==Y`在该位置处为真，否则该位置为0。

```python
X == Y
```

```
tensor([[False,  True, False,  True],
        [False, False, False, False],
        [False, False, False, False]])
```

## 1.3. 广播机制

在上面的部分中，我们看到了如何在 $相同形状$ 的两个张量上执行按元素操作。 在某些情况下，即使 $\color{red}形状不同$，我们仍然可以通过调用  *广播机制* （broadcasting mechanism）来执行按元素操作。

这种机制的工作方式如下：

- 首先，通过适当 $\color{red}复制元素$ 来扩展一个或两个数组， 以便在转换之后，两个张量具有相同的形状。
- 其次，对生成的数组执行按元素操作。

在大多数情况下，我们将沿着数组中 $\color{red}长度为1的轴进行广播$，如下例子：

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```
(tensor([[0],
         [1],
         [2]]),
 tensor([[0, 1]]))
```

由于 `a` 和 `b` 分别是 $3×1$ 和 $1×2$ 矩阵，如果让它们相加，它们的形状不匹配。 我们将两个矩阵 *broadcasting*  为一个更大的 $3×2$ 矩阵，如下所示：矩阵 `a` 将复制列， 矩阵 `b` 将复制行，然后再按元素相加。

```python
a + b
```

```
tensor([[0, 1],
        [1, 2],
        [2, 3]])
```

## 1.5. 节省内存

运行一些操作可能会导致为新结果分配内存。 例如，如果我们用 `Y=X+Y`，我们将取消引用 `Y` 指向的张量，而是指向新分配的内存处的张量。

在下面的例子中，我们用 Python 的 `id()` 函数演示了这一点， 它给我们提供了内存中引用对象的确切地址。 运行 `Y=Y+X` 后，我们会发现 `id(Y)` 指向另一个位置。 这是因为Python首先计算 `Y+X`，为结果分配新的内存，然后使 `Y` 指向内存中的这个新位置。

```
before = id(Y)
Y = Y + X
id(Y) == before
```

```
False
```

这可能是不可取的，原因有两个：

- 首先，我们不想总是不必要地分配内存。 在机器学习中，我们可能有数百兆的参数，并且在一秒内多次更新所有参数。 通常情况下，我们希望原地执行这些更新。
- 其次，如果我们不原地更新，其他引用仍然会指向旧的内存位置， 这样我们的某些代码可能会无意中引用旧的参数。

幸运的是，执行原地操作非常简单。 我们可以使用 $\color{red}切片表示法, 相当于cpp的引用$ 将操作的结果分配给先前分配的数组，例如`Y[:] = <expression>`。 为了说明这一点，我们首先创建一个新的矩阵 `Z`，其形状与另一个`Y`相同， 使用`zeros_like`来分配一个全0的块。

```
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```
id(Z): 140622147537312
id(Z): 140622147537312
```

如果在后续计算中没有重复使用 `X`， 我们也可以使用 `X[:]=X+Y` 或 `X += Y # (对于tensor来说不改变地址，但对于int来说会改变)` 来减少操作的内存开销。

```
before = id(X)
X += Y
id(X) == before
```

```
True
```