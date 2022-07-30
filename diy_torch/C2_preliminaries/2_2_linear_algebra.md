# Linear Algebra

:label:`sec_linear-algebra`



Now that you can store and manipulate data, let us briefly review the subset of basic linear algebra that you will need to understand and implement most of models covered in this book. Below, we introduce the basic mathematical objects, arithmetic, and operations in linear algebra, expressing each of them through mathematical notation and the corresponding implementation in code.

## 2.3.1. 标量



如果你曾经在餐厅支付餐费，那么你已经知道一些基本的线性代数，比如在数字间相加或相乘。 例如，北京的温度为 $52^∘F$（除了摄氏度外，另一种温度计量单位）。 严格来说，我们称仅包含一个数值的叫 *标量* （scalar）。 如果要将此华氏度值转换为更常用的摄氏度， 则可以计算表达式 $c=\frac{5}{9}(f−32)$，并将 $f$ 赋为52。 在此等式中，每一项（5、9和32）都是标量值。 符号c和f称为 *变量* （variable），它们表示未知的标量值。

在本书中，我们采用了数学表示法，其中标量变量由普通小写字母表示（例如，x、y和z）。 我们用R表示所有（连续）*实数*标量的空间。 我们之后将严格定义 *空间* （space）是什么， 但现在你只要记住表达式x∈R是表示x是一个实值标量的正式形式。 符号∈称为“属于”，它表示“是集合中的成员”。 我们可以用x,y∈{0,1}来表明x和y是值只能为0或1的数字。

标量由只有一个元素的张量表示。 在下面的代码中，我们实例化两个标量，并执行一些熟悉的算术运算，即加法、乘法、除法和指数。

```python
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

(tensor(5.), tensor(6.), tensor(1.5000), tensor(9.))
## 2.3.2. 向量



[ **You can think of a vector as simply a list of scalar values.** ] We call these values the *elements* (*entries* or  *components* ) of the vector. When our vectors represent examples from our dataset, their values hold some real-world significance. For example, if we were training a model to predict the risk that a loan defaults, we might associate each applicant with a vector whose components correspond to their income, length of employment, number of previous defaults, and other factors. If we were studying the risk of heart attacks hospital patients potentially face, we might represent each patient by a vector whose components capture their most recent vital signs, cholesterol levels, minutes of exercise per day, etc. 在数学表示法中，我们通常将向量记为粗体、小写的符号 （例如，x、y和z)）。

我们通过一维张量处理向量。一般来说，张量可以具有任意长度，取决于机器的内存限制。

```python
x = torch.arange(4)
x
```
tensor([0, 1, 2, 3])
We can refer to any element of a vector by using a subscript. For example, we can refer to the **𝑖**t**h**ith element of **𝐱**x by **𝑥**𝑖xi. Note that the element **𝑥**𝑖xi is a scalar, so we do not bold-face the font when referring to it. Extensive literature considers column vectors to be the default orientation of vectors, so does this book. In math, a vector **𝐱**x can be written as

$$
\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},

$$

其中x1,…,xn是向量的元素。在代码中，我们通过张量的索引来访问任一元素。

```python
x[3]
```
tensor(3)
### 2.3.2.1. 长度、维度和形状



向量只是一个数字数组，就像每个数组都有一个长度一样，每个向量也是如此。 在数学表示法中，如果我们想说一个向量x由n个实值标量组成， 我们可以将其表示为x∈Rn。 向量的长度通常称为向量的 *维度* （dimension）。

与普通的Python数组一样，我们可以通过调用Python的内置 `len()` 函数来访问张量的长度。

```python
len(x)
```
4
当用张量表示一个向量（只有一个轴）时，我们也可以通过 `.shape` 属性访问向量的长度。 形状（shape）是一个元素组，列出了张量沿每个轴的长度（维数）。 对于只有一个轴的张量，形状只有一个元素。

```python
x.shape
```
torch.Size([4])
请注意， *维度* （dimension）这个词在不同上下文时往往会有不同的含义，这经常会使人感到困惑。 为了清楚起见，我们在此明确一下： *向量*或*轴*的维度被用来表示*向量*或*轴*的长度，即向量或轴的元素数量。 然而，张量的维度用来表示张量具有的轴数。 在这个意义上，张量的某个轴的维数就是这个轴的长度。

## 2.3.3. 矩阵

正如向量将标量从零阶推广到一阶，矩阵将向量从一阶推广到二阶。Matrices, which we will typically denote with bold-faced, capital letters (e.g., $\mathbf{X}$, $\mathbf{Y}$, and $\mathbf{Z}$), are represented in code as tensors with two axes.

In math notation, we use $\mathbf{A} \in \mathbb{R}^{m \times n}$ to express that the matrix $\mathbf{A}$ consists of $m$ rows and $n$ columns of real-valued scalars. Visually, we can illustrate any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ as a table, where each element $a_{ij}$ belongs to the $i^{\mathrm{th}}$ row and $j^{\mathrm{th}}$ column:

$$
\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.

$$

:eqlabel:`eq_matrix_def`

For any $\mathbf{A} \in \mathbb{R}^{m \times n}$, the shape of $\mathbf{A}$ is ($m$, $n$) or $m \times n$. Specifically, when a matrix has the same number of rows and columns, its shape becomes a square; thus, it is called a *square matrix*.

We can [**create an $m \times n$ matrix**] by specifying a shape with two components $m$ and $n$ when calling any of our favorite functions for instantiating a tensor.

```python
A = torch.arange(20).reshape(5, 4)
A
```
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])
We can access the scalar element $a_{ij}$ of a matrix $\mathbf{A}$ in [(2.3.2)](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html#equation-eq-matrix-def) by specifying the indices for the row ($i$) and column ($j$), such as $[\mathbf{A}]_{ij}$. When the scalar elements of a matrix $\mathbf{A}$, such as in [(2.3.2)](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html#equation-eq-matrix-def), are not given, we may simply use the lower-case letter of the matrix $\mathbf{A}$ with the index subscript, $a_{ij}$,
to refer to $[\mathbf{A}]_{ij}$. To keep notation simple, commas are inserted to separate indices only when necessary, such as $a_{2, 3j}$ and $[\mathbf{A}]_{2i-1, 3}$. Sometimes, we want to flip the axes. When we exchange a matrix's rows and columns, the result is called the *transpose* of the matrix.

Formally, we signify a matrix $\mathbf{A}$'s transpose by $\mathbf{A}^\top$ and if $\mathbf{B} = \mathbf{A}^\top$, then $b_{ij} = a_{ji}$ for any $i$ and $j$.
Thus, the transpose of $\mathbf{A}$ in [(2.3.2)](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html#equation-eq-matrix-def) is a $n \times m$ matrix:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.

$$

Now we access a (**matrix's transpose**) in code.

```python
A.T
```
tensor([[ 0,  4,  8, 12, 16],
        [ 1,  5,  9, 13, 17],
        [ 2,  6, 10, 14, 18],
        [ 3,  7, 11, 15, 19]])
As a special type of the square matrix, [**a *symmetric matrix* $\mathbf{A}$ is equal to its transpose:
$\mathbf{A} = \mathbf{A}^\top$.**] Here we define a symmetric matrix `B`.

```python
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```
tensor([[1, 2, 3],
        [2, 0, 4],
        [3, 4, 5]])
Now we compare `B` with its transpose.

```python
B == B.T
```
tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
矩阵是有用的数据结构：它们允许我们组织具有不同模式的数据。 例如，我们矩阵中的行可能对应于不同的房屋（数据样本），而列可能对应于不同的属性。 如果你曾经使用过电子表格软件或已阅读过 [2.2节](https://zh.d2l.ai/chapter_preliminaries/pandas.html#sec-pandas)，这应该听起来很熟悉。 因此，尽管单个向量的默认方向是列向量，但在表示表格数据集的矩阵中， 将每个数据样本作为矩阵中的行向量更为常见。 我们将在后面的章节中讲到这点，这种约定将支持常见的深度学习实践。 例如，沿着张量的最外轴，我们可以访问或遍历小批量的数据样本。

## 2.3.4. 张量

Just as vectors generalize scalars, and matrices generalize vectors, we can build data structures with even more axes. [**Tensors**] (本小节中的“张量”指代数对象) (**give us a generic way of describing $n$-dimensional arrays with an arbitrary number of axes.**)

- Vectors, for example, are first-order tensors, and
- matrices are second-order tensors.
- Tensors are denoted with capital letters of a special font face (e.g., $\mathsf{X}$, $\mathsf{Y}$, and $\mathsf{Z}$) and their indexing mechanism (e.g., $x_{ijk}$ and $[\mathsf{X}]_{1, 2i-1, 3}$) is similar to that of matrices.

当我们开始处理图像时，张量将变得更加重要，图像以 $n$ 维数组形式出现， 其中3个轴对应于高度、宽度，以及一个 *通道* （channel）轴， 用于表示颜色通道（红色、绿色和蓝色）。 现在，我们先将高阶张量暂放一边，而是专注学习其基础知识。

```python
X = torch.arange(24).reshape(2, 3, 4)
X
```
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
## 2.3.5. 张量算法的基本性质

标量、向量、矩阵和任意数量轴的张量（本小节中的“张量”指代数对象）有一些实用的属性。 例如，你可能已经从按元素操作的定义中注意到，任何按元素的一元运算都不会改变其操作数的形状。 同样，给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量。 例如，将两个相同形状的矩阵相加，会在这两个矩阵上执行元素加法。

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]),
 tensor([[ 0.,  2.,  4.,  6.],
         [ 8., 10., 12., 14.],
         [16., 18., 20., 22.],
         [24., 26., 28., 30.],
         [32., 34., 36., 38.]]))
Specifically, [**elementwise multiplication 逐元素乘法 of two matrices is called their *Hadamard product 哈达玛积***] (math notation $\odot$). Consider matrix $\mathbf{B} \in \mathbb{R}^{m \times n}$ whose element of row $i$ and column $j$ is $b_{ij}$. The Hadamard product of matrices $\mathbf{A}$ (defined in [(2.3.2)](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html#equation-eq-matrix-def)) and $\mathbf{B}$

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.

$$

```python
A * B
```
tensor([[  0.,   1.,   4.,   9.],
        [ 16.,  25.,  36.,  49.],
        [ 64.,  81., 100., 121.],
        [144., 169., 196., 225.],
        [256., 289., 324., 361.]])
将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。

```python
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```
(tensor([[[ 2,  3,  4,  5],
          [ 6,  7,  8,  9],
          [10, 11, 12, 13]],
 
         [[14, 15, 16, 17],
          [18, 19, 20, 21],
          [22, 23, 24, 25]]]),
 torch.Size([2, 3, 4]))
## 2.3.6. 降维

:label:`subseq_lin-alg-reduction`

我们可以对任意张量进行的一个有用的操作是 to calculate [**the sum of their elements.**]

In mathematical notation, we express sums using the $\sum$ symbol. To express the sum of the elements in a vector $\mathbf{x}$ of length $d$, we write $\sum_{i=1}^d x_i$. In code, we can just call the function for calculating the sum.

```python
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```
(tensor([0., 1., 2., 3.]), tensor(6.))
We can express [**sums over the elements of tensors of arbitrary shape.**] For example, the sum of the elements of an $m \times n$ matrix $\mathbf{A}$ could be written $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```python
A.shape, A.sum()
```
(torch.Size([5, 4]), tensor(190.))
默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。 我们还可以指定张量沿哪一个轴来通过求和降低维度。 以矩阵为例，为了通过求和所有行的元素来降维（轴0），我们可以在调用函数时指定`axis=0`。 由于输入矩阵沿 0 轴降维以生成输出向量，因此输入轴0的维数在输出形状中消失。

```python
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```
(tensor([40., 45., 50., 55.]), torch.Size([4]))
Specifying `axis=1` will reduce the column dimension (axis 1) by summing up elements of all the columns.
Thus, the dimension of axis 1 of the input is lost in the output shape.

```python
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```
(tensor([ 6., 22., 38., 54., 70.]), torch.Size([5]))
Reducing a matrix along both rows and columns via summation
is equivalent to summing up all the elements of the matrix.

```python
A.sum(axis=[0, 1])  # Same as `A.sum()`
```
tensor(190.)
[**A related quantity is the *mean*, which is also called the *average*.**] We calculate the mean by dividing the sum by the total number of elements. In code, we could just call the function for calculating the mean on tensors of arbitrary shape.

```python
A.mean(), A.sum() / A.numel()
```
(tensor(9.5000), tensor(9.5000))
Likewise, the function for calculating the mean can also reduce a tensor along the specified axes.

```python
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```
(tensor([ 8.,  9., 10., 11.]), tensor([ 8.,  9., 10., 11.]))
### 2.3.6.1. 非降维求和

:label:`subseq_lin-alg-non-reduction`

However, sometimes it can be useful to [**keep the number of axes unchanged**] when invoking the function for calculating the sum or mean.

```python
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```
tensor([[ 6.],
        [22.],
        [38.],
        [54.],
        [70.]])
For instance, since `sum_A` still keeps its two axes after summing each row, we can (**divide `A` by `sum_A` with broadcasting.**)

```python
A / sum_A
```
tensor([[0.0000, 0.1667, 0.3333, 0.5000],
        [0.1818, 0.2273, 0.2727, 0.3182],
        [0.2105, 0.2368, 0.2632, 0.2895],
        [0.2222, 0.2407, 0.2593, 0.2778],
        [0.2286, 0.2429, 0.2571, 0.2714]])
If we want to calculate [**the cumulative sum of elements of `A` along some axis**], say `axis=0` (row by row),
we can call the `cumsum` function. This function will not reduce the input tensor along any axis.

```python
A.cumsum(axis=0)
```
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  6.,  8., 10.],
        [12., 15., 18., 21.],
        [24., 28., 32., 36.],
        [40., 45., 50., 55.]])
## Dot Products

So far, we have only performed elementwise operations, sums, and averages. And if this was all we could do, linear algebra probably would not deserve its own section. However, one of the most fundamental operations is the dot product.
Given two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, their *dot product* $\mathbf{x}^\top \mathbf{y}$ (or $\langle \mathbf{x}, \mathbf{y}  \rangle$) is a sum over the products of the elements at the same position: $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~The *dot product* of two vectors is a sum over the products of the elements at the same position~~]

```python
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```
(tensor([0., 1., 2., 3.]), tensor([1., 1., 1., 1.]), tensor(6.))
Note that
(**we can express the dot product of two vectors equivalently by performing an elementwise multiplication and then a sum:**)

```python
torch.sum(x * y)
```
tensor(6.)
Dot products are useful in a wide range of contexts.
For example, given some set of values,
denoted by a vector $\mathbf{x}  \in \mathbb{R}^d$
and a set of weights denoted by $\mathbf{w} \in \mathbb{R}^d$,
the weighted sum of the values in $\mathbf{x}$
according to the weights $\mathbf{w}$
could be expressed as the dot product $\mathbf{x}^\top \mathbf{w}$.
When the weights are non-negative
and sum to one (i.e., $\left(\sum_{i=1}^{d} {w_i} = 1\right)$),
the dot product expresses a *weighted average*.
After normalizing two vectors to have the unit length,
the dot products express the cosine of the angle between them.
We will formally introduce this notion of *length* later in this section.

## Matrix-Vector Products

Now that we know how to calculate dot products,
we can begin to understand *matrix-vector products*.
Recall the matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$
and the vector $\mathbf{x} \in \mathbb{R}^n$
defined and visualized in :eqref:`eq_matrix_def` and :eqref:`eq_vec_def` respectively.
Let us start off by visualizing the matrix $\mathbf{A}$ in terms of its row vectors

$$
\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},

$$

where each $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
is a row vector representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$.

[**The matrix-vector product $\mathbf{A}\mathbf{x}$
is simply a column vector of length $m$,
whose $i^\mathrm{th}$ element is the dot product $\mathbf{a}^\top_i \mathbf{x}$:**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.

$$

We can think of multiplication by a matrix $\mathbf{A}\in \mathbb{R}^{m \times n}$
as a transformation that projects vectors
from $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$.
These transformations turn out to be remarkably useful.
For example, we can represent rotations
as multiplications by a square matrix.
As we will see in subsequent chapters,
we can also use matrix-vector products
to describe the most intensive calculations
required when computing each layer in a neural network
given the values of the previous layer.

Expressing matrix-vector products in code with tensors, we use
the `mv` function. When we call `torch.mv(A, x)` with a matrix
`A` and a vector `x`, the matrix-vector product is performed.
Note that the column dimension of `A` (its length along axis 1)
must be the same as the dimension of `x` (its length).

```python
A.shape, x.shape, torch.mv(A, x)
```
(torch.Size([5, 4]), torch.Size([4]), tensor([ 14.,  38.,  62.,  86., 110.]))
## Matrix-Matrix Multiplication

If you have gotten the hang of dot products and matrix-vector products,
then *matrix-matrix multiplication* should be straightforward.

Say that we have two matrices $\mathbf{A} \in \mathbb{R}^{n \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$
\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.

$$

Denote by $\mathbf{a}^\top_{i} \in \mathbb{R}^k$
the row vector representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$,
and let $\mathbf{b}_{j} \in \mathbb{R}^k$
be the column vector from the $j^\mathrm{th}$ column of the matrix $\mathbf{B}$.
To produce the matrix product $\mathbf{C} = \mathbf{A}\mathbf{B}$, it is easiest to think of $\mathbf{A}$ in terms of its row vectors and $\mathbf{B}$ in terms of its column vectors:

$$
\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.

$$

Then the matrix product $\mathbf{C} \in \mathbb{R}^{n \times m}$ is produced as we simply compute each element $c_{ij}$ as the dot product $\mathbf{a}^\top_i \mathbf{b}_j$:

$$
\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.

$$

[**We can think of the matrix-matrix multiplication $\mathbf{AB}$ as simply performing $m$ matrix-vector products and stitching the results together to form an $n \times m$ matrix.**]
In the following snippet, we perform matrix multiplication on `A` and `B`.
Here, `A` is a matrix with 5 rows and 4 columns,
and `B` is a matrix with 4 rows and 3 columns.
After multiplication, we obtain a matrix with 5 rows and 3 columns.

```python
B = torch.ones(4, 3)
torch.mm(A, B)
```
tensor([[ 6.,  6.,  6.],
        [22., 22., 22.],
        [38., 38., 38.],
        [54., 54., 54.],
        [70., 70., 70.]])
Matrix-matrix multiplication can be simply called *matrix multiplication*, and should not be confused with the Hadamard product.

## Norms

:label:`subsec_lin-algebra-norms`

Some of the most useful operators in linear algebra are *norms*.
Informally, the norm of a vector tells us how *big* a vector is.
The notion of *size* under consideration here
concerns not dimensionality
but rather the magnitude of the components.

In linear algebra, a vector norm is a function $f$ that maps a vector
to a scalar, satisfying a handful of properties.
Given any vector $\mathbf{x}$,
the first property says
that if we scale all the elements of a vector
by a constant factor $\alpha$,
its norm also scales by the *absolute value*
of the same constant factor:

$$
f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).

$$

The second property is the familiar triangle inequality:

$$
f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).

$$

The third property simply says that the norm must be non-negative:

$$
f(\mathbf{x}) \geq 0.

$$

That makes sense, as in most contexts the smallest *size* for anything is 0.
The final property requires that the smallest norm is achieved and only achieved
by a vector consisting of all zeros.

$$
\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.

$$

You might notice that norms sound a lot like measures of distance.
And if you remember Euclidean distances
(think Pythagoras' theorem) from grade school,
then the concepts of non-negativity and the triangle inequality might ring a bell.
In fact, the Euclidean distance is a norm:
specifically it is the $L_2$ norm.
Suppose that the elements in the $n$-dimensional vector
$\mathbf{x}$ are $x_1, \ldots, x_n$.

[**The $L_2$ *norm* of $\mathbf{x}$ is the square root of the sum of the squares of the vector elements:**]

(**

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$**)

where the subscript $2$ is often omitted in $L_2$ norms, i.e., $\|\mathbf{x}\|$ is equivalent to $\|\mathbf{x}\|_2$. In code,
we can calculate the $L_2$ norm of a vector as follows.

```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```
tensor(5.)
In deep learning, we work more often
with the squared $L_2$ norm.

You will also frequently encounter [**the $L_1$ *norm***],
which is expressed as the sum of the absolute values of the vector elements:

(**

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

As compared with the $L_2$ norm,
it is less influenced by outliers.
To calculate the $L_1$ norm, we compose
the absolute value function with a sum over the elements.

```python
torch.abs(u).sum()
```
tensor(7.)
Both the $L_2$ norm and the $L_1$ norm
are special cases of the more general $L_p$ *norm*:

$$
\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.

$$

Analogous to $L_2$ norms of vectors,
[**the *Frobenius norm* of a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$**]
is the square root of the sum of the squares of the matrix elements:

[**

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

The Frobenius norm satisfies all the properties of vector norms.
It behaves as if it were an $L_2$ norm of a matrix-shaped vector.
Invoking the following function will calculate the Frobenius norm of a matrix.

```python
torch.norm(torch.ones((4, 9)))
```
tensor(6.)
### Norms and Objectives

:label:`subsec_norms_and_objectives`

While we do not want to get too far ahead of ourselves,
we can plant some intuition already about why these concepts are useful.
In deep learning, we are often trying to solve optimization problems:
*maximize* the probability assigned to observed data;
*minimize* the distance between predictions
and the ground-truth observations.
Assign vector representations to items (like words, products, or news articles)
such that the distance between similar items is minimized,
and the distance between dissimilar items is maximized.
Oftentimes, the objectives, perhaps the most important components
of deep learning algorithms (besides the data),
are expressed as norms.

## More on Linear Algebra

In just this section,
we have taught you all the linear algebra
that you will need to understand
a remarkable chunk of modern deep learning.
There is a lot more to linear algebra
and a lot of that mathematics is useful for machine learning.
For example, matrices can be decomposed into factors,
and these decompositions can reveal
low-dimensional structure in real-world datasets.
There are entire subfields of machine learning
that focus on using matrix decompositions
and their generalizations to high-order tensors
to discover structure in datasets and solve prediction problems.
But this book focuses on deep learning.
And we believe you will be much more inclined to learn more mathematics
once you have gotten your hands dirty
deploying useful machine learning models on real datasets.
So while we reserve the right to introduce more mathematics much later on,
we will wrap up this section here.

If you are eager to learn more about linear algebra,
you may refer to either the
[online appendix on linear algebraic operations](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html)
or other excellent resources :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`.

## Summary

* Scalars, vectors, matrices, and tensors are basic mathematical objects in linear algebra.
* Vectors generalize scalars, and matrices generalize vectors.
* Scalars, vectors, matrices, and tensors have zero, one, two, and an arbitrary number of axes, respectively.
* A tensor can be reduced along the specified axes by `sum` and `mean`.
* Elementwise multiplication of two matrices is called their Hadamard product. It is different from matrix multiplication.
* In deep learning, we often work with norms such as the $L_1$ norm, the $L_2$ norm, and the Frobenius norm.
* We can perform a variety of operations over scalars, vectors, matrices, and tensors.

## Exercises

1. Prove that the transpose of a matrix $\mathbf{A}$'s transpose is $\mathbf{A}$: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
2. Given two matrices $\mathbf{A}$ and $\mathbf{B}$, show that the sum of transposes is equal to the transpose of a sum: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
3. Given any square matrix $\mathbf{A}$, is $\mathbf{A} + \mathbf{A}^\top$ always symmetric? Why?
4. We defined the tensor `X` of shape (2, 3, 4) in this section. What is the output of `len(X)`?
5. For a tensor `X` of arbitrary shape, does `len(X)` always correspond to the length of a certain axis of `X`? What is that axis?
6. Run `A / A.sum(axis=1)` and see what happens. Can you analyze the reason?
7. When traveling between two points in Manhattan, what is the distance that you need to cover in terms of the coordinates, i.e., in terms of avenues and streets? Can you travel diagonally?
8. Consider a tensor with shape (2, 3, 4). What are the shapes of the summation outputs along axis 0, 1, and 2?
9. Feed a tensor with 3 or more axes to the `linalg.norm` function and observe its output. What does this function compute for tensors of arbitrary shape?

[Discussions](https://discuss.d2l.ai/t/31)
