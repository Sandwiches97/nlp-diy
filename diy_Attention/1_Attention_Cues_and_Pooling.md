# 1. Attention Cues

All in all, information in our environment is not scarce(稀缺), attention is.

总结：当作一个回归任务，即给定$(x_i,y_i)$，计算新加入值$x$与所有$x_i$的距离度量，然后它的attention权重则为 $f(x) = \sum_{i=1}^n \alpha(x, x_i)\cdot y_i$

## 1.1. Attention Cues in Biology

To explain how our attention is deployed in the visual world, a two-component framework has emerged and been pervasive. This idea dates back to William James in the 1890s, who is considered the “father of American psychology” [[James, 2007]](https://d2l.ai/chapter_references/zreferences.html#james-2007). In this framework, subjects selectively direct the spotlight of attention using both the **nonvolitional(非自愿的) cue** and  **volitional cue** .

The **nonvolitional cue** is based on the saliency and conspicuity （显著性） of objects in the environment. Imagine there are five objects in front of you: a newspaper, a research paper, a cup of coffee, a notebook, and a book such as in [Fig. 1.1](). While all the paper products are printed in **black and white**, the coffee cup is **red**. In other words, this coffee is intrinsically salient and conspicuous in this visual environment, automatically and involuntarily drawing attention. So you bring the fovea (the center of the macula where visual acuity is highest) onto the coffee as shown in [Fig. 1.1]().

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/eye-coffee.svg" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig 1 Using the nonvolitional cue based on saliency (red cup, non-paper), attention is involuntarily directed to the coffee.¶
  	</div>
</center>

After drinking coffee, you become caffeinated and want to read a book. So you turn your head, refocus your eyes, and look at the book as depicted in [Fig. 1.2](). Different from the case in [Fig. 1.1]() where the coffee biases you towards selecting based on saliency, in this task-dependent case you select the book **under cognitive and volitional control**. Using the **volitional cue** based on variable selection criteria, this form of attention is more deliberate. It is also more powerful **with the subject’s voluntary effort**.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/eye-book.svg" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig 1.2 Using the volitional cue (want to read a book) that is task-dependent, attention is directed to the book under volitional control.
  	</div>
</center>

## 1.2. Queries, Keys, and Values

Inspired by the nonvolitional and volitional attention cues that explain the attentional deployment, in the following we will describe a framework for designing attention mechanisms by incorporating these two attention cues.

To begin with, consider the simpler case where **only nonvolitional cues** are available. To bias selection over sensory inputs, we can simply use **a parameterized fully-connected layer** or even **non-parameterized max or average pooling**.

Therefore, what sets attention mechanisms apart from those fully-connected layers or pooling layers is the inclusion of the **volitional cues**. In the context of attention mechanisms, we refer to volitional cues as  ``queries``. Given any query, attention mechanisms bias selection over sensory inputs (e.g., intermediate feature representations) via  *attention pooling* . These sensory inputs are called ``values`` in the context of attention mechanisms. More generally, every value is paired with a  ``key`` , which can be thought of the nonvolitional cue of that sensory input. As shown in [Fig. 1.3](), we can design attention pooling so that the given query (volitional cue) can interact with keys (nonvolitional cues), which guides bias selection over values (sensory inputs).

- ``queries`` ~~ volitional cues:
  - Given any ``query``, attention mechanisms bias selection over sensory inputs (e.g., intermediate feature representations) via  *attention pooling* .
- ``keys``     ~~ nonvolitional cues
- ``values``   ~~ sensory inputs
  - sensory inputs are called ``values`` in the context of attention mechanisms. More generally, every ``value`` **is paired with** a  ``key`` , which can be thought of the nonvolitional cue of that sensory input.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/qkv.svg" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig 1.3 Attention mechanisms bias selection over values (sensory inputs) via attention pooling, which incorporates queries (volitional cues) and keys (nonvolitional cues).
  	</div>
</center>

Note that there are many alternatives for the design of attention mechanisms. For instance, we can design a non-differentiable attention model that can be trained using reinforcement learning methods [[Mnih et al., 2014]](https://d2l.ai/chapter_references/zreferences.html#mnih-heess-graves-ea-2014). Given the dominance of the framework in [Fig. 1.3](), models under this framework will be the center of our attention in this chapter.

## 1.3. Visualization of Attention

Average pooling can be treated as a weighted average of inputs, where weights are uniform. In practice, attention pooling aggregates values using weighted average, where weights are computed between the given query and different keys.

To visualize attention weights, we define the `show_heatmaps` function. Its input `matrices` has the shape (number of rows for display, number of columns for display, number of queries, number of keys).

# 2. Attention Pooling: Nadaraya-Watson Kernel Regression

Now you know the major components of attention mechanisms under the framework in [Fig. 1.3](). To recapitulate （概括）, the interactions between `queries` (volitional cues) and `keys` (nonvolitional cues) result in  **attention pooling** . The attention pooling selectively aggregates `values` (sensory inputs) to produce the output.

In this section, we will describe attention pooling in greater detail to give you a high-level view of how attention mechanisms work in practice. Specifically, the Nadaraya-Watson kernel regression model proposed in 1964 is a simple yet complete example for demonstrating machine learning with attention mechanisms.

## 2.1. Generating the Dataset

To keep things simple, let us consider the following **regression problem**: given a dataset of input-output pairs ${(x_1,y_1),…,(x_n,y_n)}$, how to learn $f$ to predict the output $\hat{y}=f(x)$ for any new input $x$?

Here we generate an artificial dataset according to the following nonlinear function with the noise term ϵ:

$$
y_i=2sin⁡(x_i)+x_i^{0.8}+ϵ, \tag{2.1}

$$

where ϵ obeys a normal distribution （高斯噪声） with zero mean and standard deviation 0.5. Both 50 training examples and 50 testing examples are generated. To better visualize the pattern of attention later, the training inputs are sorted.

The following function plots all the training examples (represented by circles), the ground-truth data generation function `f` without the noise term (labeled by “Truth”), and the learned prediction function (labeled by “Pred”).

## 2.2. Average Pooling

We begin with perhaps the world’s “dumbest” estimator for this regression problem: using average pooling to average over all the training outputs:

$$
f(x)=\frac{1}{n}\sum_{i=1}^ny_i,\tag{2.2}

$$

which is plotted below. As we can see, this estimator is indeed not so smart.

## 2.3. Nonparametric Attention Pooling

Obviously, average pooling omits the inputs $x_i$. A better idea was proposed by Nadaraya [[Nadaraya, 1964]](https://d2l.ai/chapter_references/zreferences.html#nadaraya-1964) and Watson [[Watson, 1964]](https://d2l.ai/chapter_references/zreferences.html#watson-1964) to weigh the outputs $y_i$ according to their input locations:

$$
f(x)=\sum_{i=1}^n \frac{\mathcal{K}(x−x_i)}{\sum_{j=1}^n\mathcal{K}(x−x_j)}y_i, \tag{2.3}

$$

where $\mathcal{K}$ is a  *kernel* . The estimator in [(2.3)]() is called  **`Nadaraya-Watson kernel regression`** . Here we will not dive into details of kernels. Recall the framework of attention mechanisms in [Fig. 1.3](). From the perspective of attention, we can rewrite [(2.3)]() in a more generalized form of  **attention pooling** :

$$
f(x)=\sum_{i=1}^n α(x,x_i)y_i,\tag{2.4}

$$

where $x$ is the `query` and $(x_i,y_i)$ is the `key`-`value` pair. Comparing [(2.4)]() and [(2.2)](), the attention pooling here is a **weighted average** of values $y_i$. The **attention weight** $α(x,x_i)$ in [(2.4)]() is assigned to the corresponding value $y_i$ based on the interaction between the `query` $x$ and the `key` $x_i$ modeled by $α$. For any `query`, its attention weights over all the `key`-`value` pairs are a valid probability distribution: they are non-negative and sum up to one.

To gain intuitions of attention pooling, just consider a **Gaussian kernel** defined as

$$
K(u)=\frac{1}{2π}exp⁡(−\frac{u^2}{2}). \tag{2.5}

$$

Plugging the `Gaussian kernel` into [(2.4)]() and [(2.3)]() gives

$$
\begin{aligned}
f(x)&=\sum_{i=1}^nα(x,x_i)y_i\\ \tag{2.6}
&=\sum_{i=1}^n\frac{exp⁡(−\frac{1}{2}(x−x_i)^2)}{\sum_{j=1}^n exp⁡(−\frac{1}{2}(x−x_j)^2)}y_i\\
&=\sum_{i=1}^n softmax\left(−\frac{1}{2}(x−x_i)^2\right)y_i.
\end{aligned}

$$

In [(2.6)](), a key $x_i$ that **is closer to** the given query $x$ will **get more attention** via a *larger attention weight* assigned to the key’s corresponding value $y_i$.

Notably, `Nadaraya-Watson kernel regression` is a **nonparametric model**; thus [(2.6)]() is an example of  **nonparametric attention pooling** . In the following, we plot the prediction based on this nonparametric attention model. The predicted line is smooth and closer to the ground-truth than that produced by average pooling.

## 2.4. **Parametric Attention Pooling**

Nonparametric `Nadaraya-Watson kernel` regression enjoys the **consistency (一致性)** benefit: given enough data this model converges to the optimal solution. Nonetheless, we can easily **integrate learnable parameters** into **attention pooling**.

As an example, slightly different from [(2.6)](), in the following the distance between the `query` $x$ and the `key` $x_i$ is multiplied by **a learnable parameter $W$**:

$$
\begin{aligned}\tag{2.7}
f(x)&=\sum_{i=1}^nα(x,x_i)y_i\\ 
&=\sum_{i=1}^n\frac{exp⁡(−\frac{1}{2}((x−x_i)\cdot W)^2)}{\sum_{j=1}^n exp⁡(−\frac{1}{2}((x−x_j)\cdot W)^2)}y_i\\
&=\sum_{i=1}^n softmax\left(−\frac{1}{2}((x−x_i)\cdot W)^2\right)y_i.
\end{aligned}

$$

In the rest of the section, we will train this model by learning the parameter of the attention pooling in [(2.7)]().

### 2.4.1. Batch Matrix Multiplication

To more efficiently compute attention for minibatches, we can leverage batch matrix multiplication utilities provided by deep learning frameworks.

Suppose that the first minibatch contains n matrices $X_1,…,X_n$ of shape $a×b$, and the second minibatch contains n matrices $Y_1,…,Y_n$ of shape $b×c$. Their batch matrix multiplication results in n matrices $X_1Y_1,…,X_nY_n $of shape $a×c$. Therefore, given two tensors of shape $(n, a, b)$ and $(n, b, c)$, the shape of their batch matrix multiplication output is $(n, a, c)$.

### 2.4.2. Defining the Model

Using minibatch matrix multiplication, below we define the parametric version of Nadaraya-Watson kernel regression based on the parametric attention pooling in [(2.7)]().

## 2.4.3. Training

In the following, we transform the training dataset to `keys` : $x$ and `values`: $y$ to train the `attention model`. In the parametric attention pooling, any training input takes `key-value` pairs from all the training examples except for itself to predict its output. （在参数注意力池中，任何训练input 都从**除自身之外的所有训练示例**中获取“key-value”对来预测其输出。）

Using the `squared loss` and `stochastic gradient descent`, we train the parametric attention model.
