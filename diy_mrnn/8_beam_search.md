# 8 Beam Search

:label:`sec_beam-search`

* 序列搜索策略包括 Greedy Search、Exhaustive Search 和 Beam Search。
* Greedy Search 所选取序列的计算量最小，但精度相对较低。
* Exhaustive Search 所选取序列的精度最高，但计算量最大。
* Beam Search 通过灵活选择 Beam size(束宽)，在正确率和计算代价之间进行权衡。

在 [7_seq2seq.md](7_seq2seq.md) 中，我们 $\text{\colorbox{black}{\color{yellow}逐个预测}}$ 输出序列， 直到预测序列中出现Ending-of-Sequence `“<eos>”`。 在本节中，我们将首先介绍 $\color{red}\text{\colorbox{white}{greedy search（贪心）}}$ 策略， 并探讨其存在的问题，然后对比其他替代策略：  $\color{red}\text{\colorbox{white}{exhaustive search（穷举）}}$和 $\color{red}\text{\colorbox{white}{beam search（束搜索）}}$。

Before a formal introduction to $\color{red}\text{\colorbox{white}{greedy search}}$, let us formalize the $\color{red}\text{\colorbox{black}{search problem}}$ using the same mathematical notation from [[7_seq2seq.md](7_seq2seq.md). At any time step $t'$, the probability of the decoder output $y_{t'}$ is conditional (条件概率) on the output subsequence $y_1, \ldots, y_{t'-1}$ before $t'$ and the context variable $\mathbf{c}$ that encodes the information of the input sequence.

To quantify (量化) computational cost, denote by $\color{yellow}\mathcal{Y}$ (it contains "&lt;eos&gt;") the $\text{\colorbox{black}{\color{yellow}output vocabulary}}$. So the cardinality (基数) $\left|\mathcal{Y}\right|$ of this vocabulary set is the vocabulary size. Let us also specify the maximum number of tokens of an output sequence as $\color{magenta}T'\text{\colorbox{black}{ 输出序列的最大token数}}$. As a result, our goal is to search for an ideal output from all the $复杂度：\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ possible output sequences. Of course, for all these output sequences, portions including and after "&lt;eos&gt;" will be discarded in the actual output.

## 8.1 Greedy Search

First, let us take a look at a simple strategy: $\color{red}\text{\colorbox{white}{greedy search}}$. This strategy has been used to predict sequences in [7_seq2seq.md](7_seq2seq.md).
In $\color{red}\text{\colorbox{white}{greedy search}}$, at any time step $t'$ of the output sequence, we search for the token with the highest conditional probability from $\mathcal{Y}$, i.e.,

$$
y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),

$$

as the output. Once "&lt;eos&gt;" is outputted or the output sequence has reached its maximum length $T'$, the output sequence is completed. (即，写一个$T'$ 个长度的 for循环，如果生成 eos，则直接 break)

$\color{red}\text{\colorbox{black}{So what can go wrong with greedy search}}$?
In fact, the *optimal sequence* should be the output sequence with the maximum $\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$, which is the conditional probability of generating an output sequence based on the input sequence. $\text{\colorbox{black}{\color{yellow}Unfortunately}}$, there is no guarantee that the optimal sequence will be obtained
by greedy search. 无法获得最优

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://zh.d2l.ai/_images/s2s-prob1.svg" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图9.8.1 在每个时间步，贪心搜索选择具有最高条件概率的词元¶
  	</div>
</center>

Let us illustrate it with an example. Suppose that there are four tokens `"A"`, `"B"`, `"C"`, and `"<eos>"` in the output dictionary. In [fig 9.8.1](), the four numbers under each time step represent the conditional probabilities of generating `"A"`, `"B"`, `"C"`, and `"<eos>"` at that time step, respectively. At each time step, greedy search selects the token with the highest conditional probability. Therefore, the output sequence `"A"`, `"B"`, `"C"`, and `"<eos>"` will be predicted in [fig 9.8.1]().
The conditional probability of this output sequence is $0.5\times0.4\times0.4\times0.6 = 0.048$.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://zh.d2l.ai/_images/s2s-prob2.svg" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图9.8.2 在时间步2，选择具有第二高条件概率的词元“C”（而非最高条件概率的词元）
  	</div>
</center>

Next, let us look at another example in [fig 9.8.2]().

与 [图9.8.1](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html#fig-s2s-prob1)不同，在时间步2中， 我们选择 [图9.8.2](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html#fig-s2s-prob2) 中的词元 `“C”`， 它具有*第二*高 的条件概率。 由于时间步3所基于的时间步1和2处的输出子序列已从 [图9.8.1](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html#fig-s2s-prob1) 中的 `“A”` 和 `“B”` 改变为 [图9.8.2](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html#fig-s2s-prob2)中的 `“A”`和 `“C”`， 因此时间步3处的每个词元的条件概率也在 [图9.8.2](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html#fig-s2s-prob2) 中改变。 假设我们在时间步3选择词元 `“B”`， 于是当前的时间步4基于前三个时间步的输出子序列 `“A”`、`“C”` 和 `“B”` 为条件， 这与 [图9.8.1](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html#fig-s2s-prob1)中的 `“A”`、`“B”` 和 `“C”` 不同。 因此，在 [图9.8.2](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html#fig-s2s-prob2)中的时间步4生成 每个词元的条件概率也不同于 [图9.8.1](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html#fig-s2s-prob1)中的条件概率。 结果， [图9.8.2](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html#fig-s2s-prob2)中的输出序列 `“A”`、`“C”`、`“B”`和 `“<eos>”` 的条件概率为 $0.5×0.3×0.6×0.6=0.054$， 这大于 [图9.8.1](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html#fig-s2s-prob1)中的贪心搜索的条件概率。 这个例子说明：贪心搜索获得的输出序列 “A”、“B”、“C”和“<eos>” 不一定是最佳序列。

## 8.2 Exhaustive Search 穷举 (计算量巨大)

如果目标是获得最优序列， 我们可以考虑使用 $\color{red}\text{\colorbox{white}{exhaustive search（穷举）}}$： 穷举地列举所有可能的输出序列及其条件概率， 然后计算输出条件概率最高的一个。

Although we can use exhaustive search to obtain the optimal sequence, its computational cost $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ is likely to be excessively high. For example, when $|\mathcal{Y}|=10000$ and $T'=10$, we will need to evaluate $10000^{10} = 10^{40}$ sequences. This is next to impossible! On the other hand, the computational cost of greedy search is $\mathcal{O}(\left|\mathcal{Y}\right|T')$: it is usually significantly smaller than that of exhaustive search. For example, when $|\mathcal{Y}|=10000$ and $T'=10$, we only need to evaluate $10000\times10=10^5$ sequences.

## 8.3 Beam Search

那么该选取哪种序列搜索策略呢？ 如果精度最重要，则显然是 Exhaustive Search。 如果计算成本最重要，则显然是 Greedy Search。 而 $\color{red}\text{\colorbox{white}{beam search（束搜索）}}$ 的实际应用则介于这两个极端之间。

$\color{red}\text{\colorbox{white}{beam search}}$ is an improved version of greedy search 贪心的改进版本. It has a hyperparameter named *beam size*, $k$. At time step 1, we select $k$ tokens with the highest conditional probabilities.
Each of them will be the first token of $k$ candidate output sequences, respectively. At each subsequent time step, based on the $k$ candidate output sequences at the previous time step, we continue to select $k$ candidate output sequences with the highest conditional probabilities from $k\left|\mathcal{Y}\right|$ possible choices.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://zh.d2l.ai/_images/beam-search.svg" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图9.8.3 束搜索过程（束宽：2，输出序列的最大长度：3）。候选输出序列是 A、C、AB、CE、ABD和 CED
  	</div>
</center>

[fig 9.8.3]()demonstrates the process of beam search with an example.

Suppose that the output vocabulary contains only five elements: $\mathcal{Y} = \{A, B, C, D, E\}$, where one of them is `"<eos>"`. Let the beam size be 2 and the maximum length of an output sequence be 3.

1. At time step 1, suppose that the tokens with the highest conditional probabilities $P(y_1 \mid \mathbf{c})$ are $A$ and $C$.
2. At time step 2, for all $y_2 \in \mathcal{Y},$ we compute

$$
\begin{aligned}
P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),
\end{aligned}

$$

从这十个值中选择 $\color{red}最大的两个$, say $P(A, B \mid \mathbf{c})$ and $P(C, E \mid \mathbf{c})$.

3. Then at time step 3, for all $y_3 \in \mathcal{Y}$, we compute

$$
\begin{aligned}
P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),
\end{aligned}

$$

从这十个值中选择 $\color{red}最大的两个$, say $P(A, B, D \mid \mathbf{c})$   and  $P(C, E, D \mid  \mathbf{c}).$

4. 我们会得到六个候选输出序列 (candidates output sequences): (i) $A$; (ii) $C$; (iii) $A$, $B$; (iv) $C$, $E$; (v) $A$, $B$, $D$; and (vi) $C$, $E$, $D$.
5. In the end, we obtain the set of final candidate output sequences based on these six sequences (丢弃包括 `“<eos>”` 和之后的部分).
6. 然后我们选择其中条件概率乘积最高的序列作为输出序列:

$$
\frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),

$$

where $L$ is the length of the final candidate sequence and $\alpha$ is usually set to 0.75.
因为一个较长的序列在 [(9.8.4)](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html#equation-eq-beam-search-score) 的求和中会有更多的对数项， 因此分母中的 $L^\alpha$ 用于惩罚长序列。

The computational cost of beam search is $\mathcal{O}(k\left|\mathcal{Y}\right|T')$. This result is in between that of greedy search and that of exhaustive search.

实际上，Greedy Search 可以看作是一种束宽为 1 的特殊类型的 Beam Search。 通过灵活地选择束宽，束搜索可以在正确率和计算代价之间进行权衡。

## Summary

* Sequence searching strategies include greedy search, exhaustive search, and beam search.
* Beam search provides a tradeoff between accuracy versus computational cost via its flexible choice of the beam size.

## Exercises

1. Can we treat exhaustive search as a special type of beam search? Why or why not?
2. Apply beam search in the machine translation problem in :numref:`sec_seq2seq`. How does the beam size affect the translation results and the prediction speed?
3. We used language modeling for generating text following  user-provided prefixes in :numref:`sec_rnn_scratch`. Which kind of search strategy does it use? Can you improve it?

[Discussions](https://discuss.d2l.ai/t/338)
