# 1. word2vec


- 词向量 (Word vectors) 是用于表示单词意义的向量，也可以看作是词的特征向量。将词映射到实向量的技术称为词嵌入(word embedding)。
- word2vec工具包含跳元模型 (skip-gram) 和连续词袋模型 (continue bag-of-words)。
- skip-gram假设一个单词可用于在文本序列中，生成其周围的单词；而CBOW假设基于上下文词来生成中心单词。


## 1.1 One-hot vectors are a bad choice

Suppose that the number of different words in the dictionary is `N`, and each word corresponds to a different integer (index) from 0 to N-1.

to get the one-hot vector representation for any word with index $i$, we create a length-$N$ vector with zeros, and set 1 at teh position $i$. For example, $V_i = [0, 0, ..., 1, 0, ..., 0]$. And we can calculate the similarity between vectors $x, y\in \R^d$,

$$
\frac{x^T y}{||x||\cdot||y||}\in [-1, 1] \tag{1.1}

$$

**缺点、issue:** $\color{red}\text{the similarity between any two different words is 0}$，无法表示词之间的相似性

## 1.2 self-supervised word2vec

the `word2vec` tool was proposed to address the above issue. it maps each word to a fixed-length vector, and these vectors can better express $\color{red}\text{the similarity and analogy relationship}$ among different words.

- two models
  - *skip-gram*
  - *continuous bag of words (CBOW)*

## 1.3 The Skip-Gram Model

Assumes that `a word` can be used to **generate** `its surrounding words` in a text sequence.

${\color{red}已知一个单词，生成\text{context}}$

this process can be represented by a probability expression:

$$
P("the", "man", "his", "son"\ |\ "loves") \tag{1.2}

$$

Assume that the context words are independently generated given the center word. Then Eq(1.2) can be rewritten as:

$$
P("the"|\ "loves")\cdot P("man"|\ "loves")\cdot P("his"|\ "loves")\cdot P("son"\ |\ "loves")

$$

$$
\tag{1.3}

$$

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://zh-v2.d2l.ai/_images/skip-gram.svg#pic_center" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig 1
  	</div>
</center>

In the skip-gram model, each word has two $d$-dimensional-vector representations for calculating conditional probabilities. More concretely, for two vector $v_i \in \R^d$ and $u_i \in \R^d$ , the conditional probability of generating any context word $w_o$ given the center word  $w_c$ can be modeled by a `softmax operation`: (换而言之，给定背景单词，计算某一个单词出现的概率)

$$
P(w_o | w_c) = \frac{exp(u^T_ov_c)}{\sum_{i\in V}exp(u^T_iv_c)} \tag{1.4}

$$

where the vocabulary index set $V={0, 1,...,|V|-1}$.

Given a text sequence of length $T$, where the word at time step $t$ is denoted as $w^{(t)}$. Assume that context words are independently generated given any center word. For **context window** size $m$, the likelihood function of the skip-gram model is the probability of generating all context words given any center word:

$$
\prod_{t-1}^T \prod_{-m\leq j\leq m, j\neq0}P(w^{t+j}|w^{t})\tag{1.5}

$$

where any time step that is less than 1 or greater than T can be omitted.

## 1.3.1 Training

In training, we learn the model parameters by **maximizing the likelihood function**. This is equivalent to minimizing the following loss function:

$$
-\sum^T_{t=1}\sum_{-m\leq j\leq m,\ j\neq0}logP(w^{t+j}|w^t). \tag{1.6}

$$

when using SGD to minimize the loss, in each iteration we can randomly sample a shorter subsequence to calculate the (stochastic) gradient for this subsequence to update teh model parameters.

Through differentiation, we can obtain its gradient:

$$
\begin{aligned}\tag{1.8}
\frac{\partial }{\partial v_c}logP(w_o|w_c) &= u_o-\frac{\sum_{j\in V}exp(u^T_j v_c)u_j}{\sum_{i\in V}exp(u^T_i v_c)} \\
&= u_o-\sum_{j\in V}\left(\frac{exp(u^T_j v_c)}{\sum_{i\in V}exp(u^T_i v_c)} \right)u_j \\
&= u_0 - \sum_{j\in V}P(w_j|w_c)u_j \\ 
\end{aligned}

$$

After training, for any word with index $i$ in the dictionary, we obtain both word vectors $v_i$ (as the center word) and $u_i$ (as the context word).

## 1.4 The Continuous Bag of Words (CBOW) Model

The `CBOW` model is similar to the skip-gram model. **The major difference** from the skip-gram model is that the continuous bag of words model assumes that **a center word is generated based on its surrounding context words in the text sequence.**

${\color{red}已知多个单词\text{context}，生成一个单词}$

this process can be represented by a probability expression:

$$
P("loves"\ |\ "the", "man", "his", "son") \tag{1.9}

$$

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://zh-v2.d2l.ai/_images/cbow.svg" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig 2
  	</div>
</center>

Similar with Skip model, we can calculate the conditional probability of generating any center word $w_c$ (with index $c$ in the dictionary) given its surrounding context word $w_{o1}, ..., w_{o_{2m}}$ (with index $o_1, ..., o_{2m}$ in the dictionary):

$$
P(w_c | w_{o1}, ..., w_{o_{2m}}) = \frac{exp(\frac{1}{2m}u^T_c (v_{o1}+,...,+v_{o_{2m}}))}{\sum_{i\in V}exp(u^T_i (v_{o1}+,...,+v_{o_{2m}}))} \tag{1.10}

$$

For brevity, let $W_o = \{w_{o1}, ..., w_{o_{2m}}  \}$ and $\overline v_o = \frac{1}{2m}(v_{o1}+,...,+v_{o_{2m}}) $. Then the (1.10) can be simplified as

$$
P(w_c|W_o) = \frac{exp(u^T_o \overline v_o)}{\sum_{i\in V}exp(u^T_i \overline v_o)} \tag{1.11}

$$

Given a text sequence of length $T$, where the word at time step $t$ is denoted as $w^{(t)}$. Assume that context words are independently generated given any center word. For **context window** size $m$, the likelihood function of the skip-gram model is the probability of generating all context words given any center word:

$$
\prod_{t-1}^T P(w^{t}|w^{t-m},...,w^{t-1},w^{t+1},...,w^{t+m})\tag{1.12}

$$

## 1.4.1 Training

Similar to 1.3.1, we can obtain its gradient:

$$
\begin{aligned}\tag{1.13} 
\frac{\partial }{\partial v_{o_i}}logP(w_o|W_o) &= \frac{1}{2m}\left(u_o-\sum_{j\in V}\left(\frac{exp(u^T_j \overline v_o)}{\sum_{i\in V}exp(u^T_i \overline v_o)} \right)u_j\right) \\
&= \frac{1}{2m}\left(u_0 - \sum_{j\in V}P(w_j|W_o)u_j \right) \\
\end{aligned}

$$
