# 3. 延后初始化 tensorflow

到目前为止，我们忽略了建立网络时需要做的以下这些事情：

* 我们定义了网络架构，但没有指定输入维度。
* 我们添加层时没有指定前一层的输出维度。
* 我们在初始化参数时，甚至没有足够的信息来确定模型应该包含多少参数。

你可能会对我们的代码能运行感到惊讶。 毕竟，深度学习框架无法判断网络的输入维度是什么。 这里的诀窍是框架的 *延后初始化* （defers initialization）， 即直到数据第一次通过模型传递时，框架才会动态地推断出每个层的大小。

在以后，当使用卷积神经网络时， 由于输入维度（即图像的分辨率）将影响每个后续层的维数， 有了该技术将更加方便。 现在我们在编写代码时无须知道维度是什么就可以设置参数， 这种能力可以大大简化定义和修改模型的任务。 接下来，我们将更深入地研究初始化机制。

## 5.3.1. 实例化网络

首先，让我们实例化一个多层感知机。

```python
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```
