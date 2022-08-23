## 一、为什么使用Pointer Network？

  传统的seq2seq模型是无法解决 $\text{\colorbox{black}{\color{yellow}输出序列}}$ 的 $\text{\colorbox{black}{\color{yellow}词汇表}}$ 会 $\color{red}随着输入序列长度的改变而改变$ 的问题的（解空间固定）。Pointer Network可以通过给输入序列的元素予一个指针，从而使得解空间不固定，可以解决OOV问题。总结来说，传统的seq2seq模型会要求有一个固定空间的大小，如果我们从不同两者之间做维度的切换（解空间发生变化时），就会出现 OOV问题。

如寻找凸包等。因为对于这类问题，输出往往是输入集合的子集。基于这种特点，作者考虑能不能找到一种结构类似编程语言中的指针，每个指针对应输入序列的一个元素， 从而我们可以直接操作输入序列而不需要特意设定输出词汇表 。


## 二、Pointer Network的结构

基于 RNN-based 的 attention 结构，the context variable at the decoding time step $t^′$ is the output of attention pooling:

$$
c_t^′=\sum_{t=1}^T{\color{red}\alpha}(s_{t^′−1},h_t)h_t,

$$

where the **decoder** hidden state $s_{t^′−1}$ at time step $t^′−1$ is the `query`, and the **encoder** hidden states $h_t$ are both the `keys` and `values`, and the attention weight $\color{red}\alpha$ is computed as in

$$
{\color{red}α}(\textbf{q},\textbf{k}_i)=softmax({\color{magenta}a}(\textbf{q},\textbf{k}_i))=\frac{exp⁡({\color{magenta}a}(\textbf{q},\textbf{k}_i))}{\sum_{j=1}^m exp⁡({\color{magenta}a}(\textbf{q},\textbf{k}_j))}\in \mathbb{R}. \tag{3.2}

$$

where $a$ is the **additive attention scoring function**:

$$
{\color{magenta}a}(q,k)=w_v^⊤tanh(W_q q + W_k k)\in \mathbb{R},\tag{3.3}

$$




下图是一个 RNN-based 的seq2seq 结构，其所输出的单词，来源于输入序列的词表

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://img-blog.csdnimg.cn/1d0fe61f06674864b174f5a732a8b1e8.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70#pic_center" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 指针网络示意图
  	</div>
</center>

由传统Attention公式可以得到，以Decoder层的第一个隐状态(标的物)为例，对于Encoder层的隐状态都有一个权重 ${\color{red}\alpha_j^1}\text{，其中 } j\in\{1, 2, ..., T\}$，指针指向权重最大的点即会把 $\color{red}权重最大的$ 作为当前的输出，可以将这个输出作为 Decoder 中下一个神经元的输入，这就是为Pointer Network。

### 2.1 Pointer Network网络如何解决OOV问题

  假设传统的seq2seq模型，在之前的场景中，词典大小为50维，如果来到一个新的场景，输入的词典大小为10000维，那么在softmax中剩余的9500个词都是OOV。
  假设使用的是Pointer Network网络，输入时10000维，每次只需要在输入中找就可以，不再建立词典，也不需要做softmax从而映射到OOV。
  总结来说，

- 传统seq2seq模型是从词表里面挑，需要做softmax；
- 而Pointer Network网络是从输入中挑，不需要做softmax。

## 三、如何通过结合Pointer Network处理语言生成?

   上面已经介绍了Pointer Network ，那么如何通过结合Pointer Network处理语言生成呢?

- Language Model是自由，灵活，不可控；
- Pointer Net 是相对可控，信息来源于输入的信息范围；
  - Pointer Net是天生的复制粘贴利器，这是一种抽取的方式。
  - 抽取式比较死板，

所以，我们可以利用 抽取 与language Model 结合到一起得到更好的生成的网络。



下面通过《[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)》这篇文章来学习如何通过结合Pointer Network处理语言生成的。
 

综合 Pointer Network 与 language generation Model

- language generation Model
  生成时会有Context Vector，然后将Context Vector投影到Vocabulary distribution上面去
- Pointer Network
  在上面的Pointer Network中，我们选取的是attention weight的最大值对应的输入作为输出；在这篇论文中，我们选取的是attention distribution。


<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://img-blog.csdnimg.cn/3f105beae119461c88ae6d077e46df60.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70" width = "70%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. PNG
  	</div>
</center>

综合Pointer Network与language generation Model的关键是将Vocabulary distribution与attention distribution进行加权平均，权重分别为 $1-p_{gen}$ 与 $p_{gen}$。假设 the context vector $h^*_t$, the decoder state $s_t$ and the decoder input $x_t$:

$$
p_{gen} = \sigma(w^T_{h^*}h^*_t + w^T_s s_t + w^T_x x_t + b_{ptr})

$$

where vectors $w_{h^*}, w_{s}, w_{x}$ and scalar $b_{ptr}$ are learnable parameters and $\sigma$ is the sigmoid function.

因此，我们可以得到第 t step 的综合单词生成概率分布：

$$
P(w) = p_{gen}P_{vocab}(w) + (1-p_{gen})\sum_{i:w_i=w}{\color{magenta}a}^t_i

$$

Note that if $w$ is an out-of-vocabulary (OOV) word, then $P_{vocab}(w)$ is zero;

similarly if $w$ does not appear in the source document, then $\sum_{i:w_i=w } {\color{magenta}a}_i^t$ is zero.
