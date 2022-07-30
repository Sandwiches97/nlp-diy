# Recurrent Neural Networks

:label:`sec_rnn`

- 对隐状态使用循环计算的神经网络称为循环神经网络（RNN）。
- 循环神经网络的**隐状态**，可以捕获**直到当前时间步**序列的历史信息。
- 循环神经网络模型的**参数数量**，不会随着时间步的增加而增加。
- 我们可以使用循环神经网络创建字符级语言模型。
- 我们可以使用**困惑度** Perplexity 来评价语言模型的质量。

* “通过时间反向传播”仅仅适用于反向传播在具有隐状态的序列模型。
* 截断是计算方便性和数值稳定性的需要。截断包括：规则截断和随机截断。
* $\color{red}矩阵的高次幂$ 可能导致 神经网络特征值的 (>1) 发散或 (<1) 消失，将以梯度爆炸或梯度消失的形式表现。
* 为了计算的效率，“通过时间反向传播”在计算期间会缓存中间值。

In [Section 8.3](https://d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html#sec-language-model), we introduced **$n$-gram models**, where the conditional probability of word $x_t$ at time step $t$ only depends on the $n-1$ previous words.

If we want to incorporate the possible effect of words earlier than time step $t-(n-1)$ on $x_t$, we need to increase $n$. **However**, the number of model parameters would also increase exponentially with it, as we need to store $|\mathcal{V}|^n$ numbers for a vocabulary set $\mathcal{V}$. **Hence**, rather than modeling $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$ it is preferable to use a latent variable model:

$$
P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}), \tag{8.4.1}

$$

where $h_{t-1}$ is a *hidden state* (also known as a hidden variable) that **stores the sequence information** up to time step $t-1$.($h_t$ 用来存序列信息) In general, the hidden state at any time step $t$ could be computed based on both the current input $x_{t}$ and the previous hidden state $h_{t-1}$:

$$
h_t = f(x_{t}, h_{t-1}). \tag{8.4.2}

$$

For a sufficiently powerful function $f$ in [(8.4.2)](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#equation-eq-ht-xt),the **latent variable model** is not an approximation. After all, $h_t$ may simply store all the data it has observed so far. **However**, it could potentially make both computation and storage expensive.

Recall that we have discussed hidden layers with hidden units in [Section 4](https://d2l.ai/chapter_multilayer-perceptrons/index.html#chap-perceptrons). It is noteworthy that hidden layers and hidden states refer to two very different concepts.

- **Hidden layers** are, as explained, layers that are hidden from view on the path from input to output.
- **Hidden states** are technically speaking **inputs** to whatever we do at a given step, and they can only be computed by looking at data at previous time steps.

***Recurrent neural networks*** (RNNs) are neural networks with hidden states. Before introducing the RNN model, we first revisit the MLP model introduced in [Section 4.1](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html#sec-mlp).

## Neural Networks without Hidden States （RECALL）

Let us take a look at an MLP with a single hidden layer. Let the hidden layer's `activation function` be $\phi$. Given a minibatch of examples $\mathbf{X} \in \mathbb{R}^{n \times d}$ with `batch size` $n$ and $d$ inputs, the hidden layer's output $\mathbf{H} \in \mathbb{R}^{n \times h}$ is calculated as：

$$
\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h). \tag{8.4.3}

$$

In [(8.4.3)](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#equation-rnn-h-without-state), we have the weight parameter $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$, the bias parameter $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$, and the number of hidden units $h$, for the hidden layer.
Thus, broadcasting (see [Section 2.1.3](https://d2l.ai/chapter_preliminaries/ndarray.html#subsec-broadcasting)) is applied during the summation. Next, the hidden variable $\mathbf{H}$ is used as the input of the output layer. The output layer is given by

$$
\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,

$$

where $\mathbf{O} \in \mathbb{R}^{n \times q}$ is the output variable, $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ is the weight parameter, and $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ is the bias parameter of the output layer.  If it is a classification problem, we can use $\text{softmax}(\mathbf{O})$ to compute the probability distribution of the output categories.

This is entirely analogous to the **regression problem** we solved previously in [Section 8.1](https://d2l.ai/chapter_recurrent-neural-networks/sequence.html#sec-sequence), hence we omit details. Suffice it to say that we can pick feature-label pairs at random and learn the parameters of our network via automatic differentiation and stochastic gradient descent.

## Recurrent Neural Networks with Hidden States

:label:`subsec_rnn_w_hidden_states`

Matters are entirely different when we have hidden states. Let us look at the structure in some more detail.

Assume that we have a minibatch of inputs $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ at time step $t$. In other words, for a minibatch of $n$ sequence examples, each row of $\mathbf{X}_t$ corresponds to one example at time step $t$ from the sequence. Next, denote by $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$ the hidden variable of time step $t$. Unlike the MLP, here we save the hidden variable $\mathbf{H}_{t-1}$ from the previous time step and introduce a new weight parameter $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ to describe how to use the hidden variable of the previous time step in the current time step. Specifically, the calculation of the hidden variable of the current time step is determined by the input of the current time step together with the hidden variable of the previous time step:

$$
\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + {\color{red}\mathbf{H}_{t-1} \mathbf{W}_{hh}}  + \mathbf{b}_h). \tag{8.4.5}

$$

Compared with [(8.4.3)](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#equation-rnn-h-without-state), [(8.4.5)](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#equation-rnn-h-with-state) adds one more term ${\color{red}\mathbf{H}_{t-1} \mathbf{W}_{hh}}$ and thus instantiates [(8.4.2)](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#equation-eq-ht-xt). From the relationship between hidden variables $\mathbf{H}_t$ and $\mathbf{H}_{t-1}$ of adjacent time steps, we know that these variables captured and retained the sequence's $\color{red}\text{historical information}$ up to their current time step, just like the state or memory of the neural network's current time step. Therefore, such a hidden variable is called a $\color{red}\text{hidden state}$. Since the hidden state uses the same definition of the previous time step in the current time step, the computation of [(8.4.5)](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#equation-rnn-h-with-state) is ***recurrent***. Hence, neural networks with hidden states based on recurrent computation are named ***recurrent neural networks***. Layers that perform the computation of [(8.4.5)](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#equation-rnn-h-with-state) in RNNs are called ***recurrent layers***.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://zh.d2l.ai/_images/rnn-bptt.svg" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图8.7.2 上图表示具有三个时间步的循环神经网络模型依赖关系的计算图。未着色的方框表示变量，着色的方框表示参数，圆表示运算符¶
  	</div>
</center>

Parameters of the RNN include the weights $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, {\color{red}\mathbf{W}_{hh}} \in \mathbb{R}^{h \times h}$, and the bias $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ of the $\color{red}\text{hidden layer}$, together with the weights $\color{magenta}\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ and the bias $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ of the $\color{magenta}\text{output layer}$. It is worth mentioning that even at different time steps, RNNs always use these model parameters.
Therefore, the parameterization cost of an RNN does not grow as the number of time steps increases.

[Fig. 8.4.1](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#fig-rnn)  illustrates the computational logic of an RNN at three adjacent time steps. At any time step $t$, the computation of the hidden state can be treated as:

- (i) concatenating the input $\mathbf{X}_t$ at the current time step $t$ and the hidden state $\mathbf{H}_{t-1}$ at the previous time step $t-1$;
- (ii) feeding the concatenation result into a fully-connected layer with the activation function $\phi$.

The output of such a fully-connected layer is the hidden state $\mathbf{H}_t$ of the current time step $t$. In this case, the model parameters are the concatenation of $\mathbf{W}_{xh}$ and $\mathbf{W}_{hh}$, and a bias of $\mathbf{b}_h$, all from :eqref:`rnn_h_with_state`. The hidden state of the current time step $t$, $\mathbf{H}_t$, will participate in computing the hidden state $\mathbf{H}_{t+1}$ of the next time step $t+1$. What is more, $\mathbf{H}_t$ will also be fed into the fully-connected output layer to compute the output
$\mathbf{O}_t$ of the current time step $t$.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/rnn.svg" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 8.4.1 An RNN with a hidden state
  	</div>
</center>

We just mentioned that the calculation of $\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$ for the hidden state is equivalent to matrix multiplication of concatenation of $\mathbf{X}_t$ and $\mathbf{H}_{t-1}$ and concatenation of $\mathbf{W}_{xh}$ and $\mathbf{W}_{hh}$. Though this can be proven in mathematics, in the following we just use a simple code snippet to show this.
To begin with, we define matrices `X`, `W_xh`, `H`, and `W_hh`, whose shapes are (3, 1), (1, 4), (3, 4), and (4, 4), respectively. Multiplying `X` by `W_xh`, and `H` by `W_hh`, respectively, and then adding these two multiplications, we obtain a matrix of shape (3, 4).

```python
import torch
from d2l import torch as d2l
```

```python
X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
```

tensor([[ 1.4019, -1.5276,  4.5853, -0.2790],
[-4.3955, -2.4508, -0.4531,  1.3304],
[-0.3181,  1.3874, -1.7398, -1.2179]])
Now we concatenate the matrices `X` and `H` along columns (axis 1), and the matrices `W_xh` and `W_hh` along rows (axis 0). These two concatenations result in matrices of shape (3, 5) and of shape (5, 4), respectively. Multiplying these two concatenated matrices, we obtain the same output matrix of shape (3, 4) as above.

```python
torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
```

tensor([[ 1.4019, -1.5276,  4.5853, -0.2790],
[-4.3955, -2.4508, -0.4531,  1.3304],
[-0.3181,  1.3874, -1.7398, -1.2179]])

## RNN-based Character-Level Language Models 字符级别

Recall that for language modeling in [Section 8.3](https://d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html#sec-language-model), we aim to predict the next token based on the current and past tokens, thus we shift the original sequence by one token as the labels. Bengio et al. first proposed to use a neural network for language modeling [[Bengio et al., 2003]](https://d2l.ai/chapter_references/zreferences.html#bengio-ducharme-vincent-ea-2003). In the following we illustrate how RNNs can be used to build a language model. Let the minibatch size be one, and the sequence of the text be “machine”.

To simplify training in subsequent sections, we tokenize text into characters rather than words and consider a  *character-level language model* . [Fig. 8.4.2](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#fig-rnn-train) demonstrates how to predict the next character based on the current and previous characters via an RNN for character-level language modeling.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/rnn-train.svg" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 8.4.2 A character-level language model based on the RNN. The input and label sequences are “machin” and “achine”, respectively.¶
  	</div>
</center>

During the training process, we run a softmax operation on the output from the output layer for each time step, and then use the cross-entropy loss to compute the error between the model output and the label. Due to the recurrent computation of the hidden state in the hidden layer, the output of time step 3 in [Fig. 8.4.2](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#fig-rnn-train), $O_3$, is determined by the text sequence “m”, “a”, and “c”. Since the next character of the sequence in the training data is “h”, the loss of time step 3 will depend on the probability distribution of the next character generated based on the feature sequence “m”, “a”, “c” and the label “h” of this time step.

In practice, each token is represented by a d-dimensional vector, and we use a batch size $n>1$. Therefore, the input $X_t$ at time step $t$ will be a $n×d$ matrix, which is identical to what we discussed in [Section 8.4.2](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#subsec-rnn-w-hidden-states).

## Perplexity 困惑度

**概要：**

- 用来度量一个概率分布或概率模型预测样本的好坏程度，如下：

![](https://pic1.zhimg.com/80/v2-da6384d62d15cd61e36d5749ff127670_720w.jpg)

**正文：**

Last, let us discuss about **how to measure the language model quality**, which will be used to evaluate our RNN-based models in the subsequent sections. One way is to check how surprising the text is. A good language model is able to predict with high-accuracy tokens that what we will see next.

Consider the following continuations of the phrase “It is raining”, as proposed by different language models:

1. “It is raining outside”
2. “It is raining banana tree”
3. “It is raining piouw;kcj pwepoiut”

In terms of quality,

- example 1 is clearly the best. The words are sensible and logically coherent. **While** it might not quite accurately reflect which word follows semantically (“in San Francisco” and “in winter” would have been perfectly reasonable extensions), the model is able to capture which kind of word follows.
- Example 2 is considerably worse by producing a nonsensical extension. **Nonetheless**, at least the model has learned how to spell words and some degree of correlation between words.
- Last, example 3 indicates a poorly trained model that does not fit data properly.

We might **measure** the quality of the model **by computing the likelihood of the sequence.** 句子的似然概率

**Unfortunately** this is a number that **is hard to understand and difficult to compare**. After all, shorter sequences are much more likely to occur than the longer ones （较短的句子出现的频率大，因此模型也喜欢生成短句子）, hence evaluating the model on Tolstoy’s magnum opus *< War and Peace >* will inevitably produce a much smaller likelihood than, say, on Saint-Exupery’s novella  *< The Little Prince >* . What is missing is the equivalent of an average.

**Information theory** comes handy here. We have defined entropy, surprisal, and cross-entropy when we introduced the softmax regression ([Section 3.4.7](https://d2l.ai/chapter_linear-networks/softmax-regression.html#subsec-info-theory-basics)) and more of information theory is discussed in the [online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html). If we want to compress text, we can ask about predicting the next token given the current set of tokens. **A better language model** should allow us to predict the next token more accurately. **Thus**, it should allow us to spend **fewer bits in compressing the sequence**. So we can measure it by the cross-entropy loss averaged **over all the n tokens of a sequence**:

$$
\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1), \tag{8.4.7}

$$

where $P$ is given by **a language model** and $x_t$ is the actual token observed at time step $t$ from the sequence. This makes the performance on documents of different lengths comparable. For historical reasons, scientists in NLP prefer to use a quantity called  $\text{\colorbox{white}{\color{red}perplexity}}$ . In a nutshell, it is the exponential of [(8.4.7)](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#equation-eq-avg-ce-for-lm):

$$
\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right). \tag{8.4.8}

$$

**Perplexity** can be best understood as the harmonic mean of the number of real choices that we have when deciding which token to pick next. Let us look at a number of cases:

* In the **best case** scenario, the model always perfectly estimates the probability of the label token as 1.
  * In this case the **perplexity** of the model is `perplexity=1`.
* In the **worst case** scenario, the model always predicts the probability of the label token as 0.
  * In this situation, the **perplexity** is positive infinity （正无穷）`perplexity= +∞`.
* At the **baseline**, the model predicts **a uniform distribution** over all the available tokens of the vocabulary.
  * In this case, the perplexity equals the number of unique tokens of the vocabulary `perplexity= 词库中unique tokens的数量`. In fact, if we were to store the sequence without any compression, this would be the best we could do to encode it. **Hence**, this provides a nontrivial upper bound that any useful model must beat.

In the following sections, we will implement RNNs for character-level language models and use perplexity to evaluate such models.

## Summary

* A neural network that uses recurrent computation for hidden states is called a recurrent neural network (RNN).
* The hidden state of an RNN can capture historical information of the sequence up to the current time step.
* The number of RNN model parameters does not grow as the number of time steps increases.
* We can create character-level language models using an  RNN.
* We can use perplexity to evaluate the quality of language models.

## Exercises

1. If we use an RNN to predict the next character in a text sequence, what is the required dimension for any output?
2. Why can RNNs express the conditional probability of a token at some time step based on all the previous tokens in the text sequence?
3. What happens to the gradient if you backpropagate through a long sequence?
4. What are some of the problems associated with the language model described in this section?

[Discussions](https://discuss.d2l.ai/t/1050)
