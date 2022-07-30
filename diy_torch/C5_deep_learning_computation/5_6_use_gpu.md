# 6 GPUs

:label:`sec_use_gpu`

* 我们可以指定用于存储和计算的设备，例如CPU或GPU。默认情况下，数据在主内存中创建，然后使用CPU进行计算。
* 深度学习框架要求计算的所有输入数据都在同一设备上，无论是CPU还是GPU。
* 不经意地移动数据可能会显著降低性能。一个典型的错误如下：计算GPU上每个小批量的损失，并在命令行中将其报告给用户（或将其记录在NumPy `ndarray`中）时，将触发全局解释器锁，从而使所有GPU阻塞。最好是为GPU内部的日志分配内存，并且只移动较大的日志。


在 [1.5节](https://zh.d2l.ai/chapter_introduction/index.html#tab-intro-decade)中， 我们回顾了过去20年计算能力的快速增长。 简而言之，自2000年以来，GPU性能每十年增长1000倍。

本节，我们将讨论如何利用这种计算性能进行研究。 首先是如何使用单个GPU，然后是如何使用多个GPU和多个服务器（具有多个GPU）。

我们先看看如何使用单个NVIDIA GPU进行计算。 首先，确保你至少安装了一个NVIDIA GPU。 然后，下载[NVIDIA驱动和CUDA](https://developer.nvidia.com/cuda-downloads) 并按照提示设置适当的路径。 当这些准备工作完成，就可以使用`<span class="pre">nvidia-smi</span>`命令来查看显卡信息。

```python
!nvidia-smi
```

Thu Mar 24 11:49:05 2022     
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.27.04    Driver Version: 460.27.04    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:1B.0 Off |                    0 |
| N/A   52C    P0    45W / 300W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  Off  | 00000000:00:1C.0 Off |                    0 |
| N/A   41C    P0    51W / 300W |   1546MiB / 16160MiB |      8%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  Off  | 00000000:00:1D.0 Off |                    0 |
| N/A   41C    P0    52W / 300W |   1702MiB / 16160MiB |     10%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   43C    P0    45W / 300W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                             
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    1   N/A  N/A     83275      C   ...l-en-release-0/bin/python     1543MiB |
|    2   N/A  N/A     83275      C   ...l-en-release-0/bin/python     1699MiB |
+-----------------------------------------------------------------------------+
在PyTorch中，每个数组都有一个设备（device）， 我们通常将其称为上下文（context）。 默认情况下，所有变量和相关的计算都分配给CPU。Typically, other contexts might be various GPUs。 当我们跨多个服务器部署作业时，事情会变得更加棘手。 通过智能地将数组分配给上下文， 我们可以最大限度地减少在设备之间传输数据的时间。 例如，当在带有GPU的服务器上训练神经网络时， 我们通常希望模型的参数在GPU上。

Next, we need to confirm that the GPU version of PyTorch is installed. If a CPU version of PyTorch is already installed, we need to uninstall it first. For example, use the `pip uninstall torch` command,
then install the corresponding PyTorch version according to your CUDA version. Assuming you have CUDA 10.0 installed, you can install the PyTorch version that supports CUDA 10.0 via `pip install torch-cu100`.

要运行此部分中的程序，至少需要两个GPU。 注意，对于大多数桌面计算机来说，这可能是奢侈的，但在云中很容易获得。 例如，你可以使用 AWS EC2 multi-GPU 实例。 本书的其他章节大都不需要多个GPU， 而本节只是为了展示数据如何在不同的设备之间传递。

## 6.1 [**Computing Devices**]

我们可以指定用于存储和计算的 device，如CPU和GPU。 默认情况下，张量是在内存中创建的，然后使用CPU计算它。

In PyTorch, the CPU and GPU can be indicated by `torch.device('cpu')` and `torch.device('cuda')`.
It should be noted that：

- the `cpu` device means all physical CPUs and memory. This means that PyTorch's calculations will try to use all CPU cores.
- However, a `gpu` device only represents one card and the corresponding memory.
- If there are multiple GPUs, we use `torch.device(f'cuda:{i}')` to represent the $i^\mathrm{th}$ GPU ($i$ starts from 0). Also, `gpu:0` and `gpu` are equivalent.

```python
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
```
(device(type='cpu'), device(type='cuda'), device(type='cuda', index=1))
我们可以查询可用gpu的数量。

```python
torch.cuda.device_count()
```
2
现在我们定义了两个方便的函数， 这两个函数允许我们在不存在所需所有GPU的情况下运行代码。

```python
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```
(device(type='cuda', index=0),
 device(type='cpu'),
 [device(type='cuda', index=0), device(type='cuda', index=1)])
## 6.2 Tensors and GPUs

我们可以查询张量所在的设备。 默认情况下，张量是在CPU上创建的。

```python
x = torch.tensor([1, 2, 3])
x.device
```
device(type='cpu')
It is important to note that whenever we want to operate on multiple terms, 它们都必须在同一个设备上.
例如，如果我们对两个张量求和， 我们需要确保两个张量都 $\color{red}位于同一个设备上$， 否则框架将不知道在哪里存储结果，甚至不知道在哪里执行计算。

### 6.2.1 Storage on the GPU

有几种方法可以在GPU上存储张量。 例如，我们可以在创建张量时指定存储设备。接 下来，我们在第一个`GPU` 上创建张量变量 `X`。 在GPU上创建的张量只消耗这个GPU的显存。

我们可以使用`nvidia-smi`命令查看显存使用情况。 一般来说，我们需要确保不创建超过GPU显存限制的数据。

```python
X = torch.ones(2, 3, device=try_gpu())
X
```
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
假设你至少有两个 GPU，下面的代码将在第二个 GPU 上创建一个随机张量。

```python
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```
tensor([[0.6826, 0.2817, 0.1645],
        [0.3779, 0.1049, 0.3581]], device='cuda:1')
### 6.2.2 Copying

[**If we want to compute `X + Y`, 我们需要决定在哪里执行这个操作.**]
For instance，如 [图5.6.1](https://zh.d2l.ai/chapter_deep-learning-computation/use-gpu.html#fig-copyto)所示，we can transfer `X` to the $\color{red}\text{second GPU}$ and perform the operation there.
不要简单地`X`加上`Y`，因为这会导致异常. The runtime engine would not know what to do: it cannot find data on $\color{red}\text{the same device}$ and it fails.

Since `Y` lives on the second GPU, we need to move `X` there before we can add the two.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://zh.d2l.ai/_images/copyto.svg" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图5.6.1 复制数据以在同一设备上执行操作
  	</div>
</center>

```python
Z = X.cuda(1)
print(X)
print(Z)
```
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:1')
现在数据在同一个GPU上（`Z`和`Y`都在），我们可以将它们相加。 

```python
Y + Z
```
tensor([[1.6826, 1.2817, 1.1645],
        [1.3779, 1.1049, 1.3581]], device='cuda:1')
假设变量`Z`已经存在于第二个GPU上。 如果我们还是调用`Z.cuda(1)`会发生什么？ 它将返回`Z`，而不会复制并分配新内存。

```python
Z.cuda(1) is Z
```
True
### 6.2.3 Side Notes 旁注

人们使用GPU来进行机器学习，因为单个GPU相对运行速度快。 但是在设备（CPU、GPU和其他机器）之间传输数据比计算慢得多。So we want you to be 100% certain that you want to do something slow before we let you do it. 如果深度学习框架只是自动复制而不会崩溃，那么你可能不会意识到你写了一些缓慢的代码。

此外，在设备（CPU、GPU和其他机器）之间传输数据要比计算慢得多。这也使得并行化变得更加困难，因为我们必须等待数据被发送（或者更确切地说是被接收），然后才能继续进行更多的操作。这就是为什么拷贝操作应该非常小心的原因。根据经验，许多小手术比一次大手术要糟糕得多。此外，除非您知道自己在做什么，否则一次执行几个操作要比代码中分散的许多单个操作好得多。这是因为如果一个设备必须等待另一个设备才能执行其他操作，则此类操作可能会阻塞。这有点像排队点咖啡，而不是通过电话预订咖啡，然后发现咖啡已经准备好了。

最后，当我们打印张量或将张量转换为NumPy格式时， 如果数据不在内存中，框架会首先将其复制到内存中， 这会导致额外的传输开销。 更糟糕的是，它现在受制于全局解释器锁，使得一切都得等待Python完成。

## 6.3 [**Neural Networks and GPUs**]

类似地，神经网络模型可以指定设备。 下面的代码将模型参数放在GPU上。

```python
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```
在接下来的几章中， 我们将看到更多关于如何在GPU上运行模型的例子， 因为它们将变得更加计算密集。

当输入为GPU上的张量时，模型将在同一GPU上计算结果。

```python
net(X)
```
tensor([[0.1809],
        [0.1809]], device='cuda:0', grad_fn=<AddmmBackward0>)
让我们确认模型参数存储在同一个GPU上。

```python
net[0].weight.data.device
```
device(type='cuda', index=0)
总之，只要所有的数据和参数都在同一个设备上， 我们就可以有效地学习模型。 在下面的章节中，我们将看到几个这样的例子。

## Summary

* We can specify devices for storage and calculation, such as the CPU or GPU.
  By default, data are created in the main memory
  and then use the CPU for calculations.
* The deep learning framework requires all input data for calculation
  to be on the same device,
  be it CPU or the same GPU.
* You can lose significant performance by moving data without care.
  A typical mistake is as follows: computing the loss
  for every minibatch on the GPU and reporting it back
  to the user on the command line (or logging it in a NumPy `ndarray`)
  will trigger a global interpreter lock which stalls all GPUs.
  It is much better to allocate memory
  for logging inside the GPU and only move larger logs.

## Exercises

1. Try a larger computation task, such as the multiplication of large matrices,
   and see the difference in speed between the CPU and GPU.
   What about a task with a small amount of calculations?
2. How should we read and write model parameters on the GPU?
3. Measure the time it takes to compute 1000
   matrix-matrix multiplications of $100 \times 100$ matrices
   and log the Frobenius norm of the output matrix one result at a time
   vs. keeping a log on the GPU and transferring only the final result.
4. Measure how much time it takes to perform two matrix-matrix multiplications
   on two GPUs at the same time vs. in sequence
   on one GPU. Hint: you should see almost linear scaling.

[Discussions](https://discuss.d2l.ai/t/63)
