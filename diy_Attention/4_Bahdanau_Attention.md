# 4. Bahdanau Attention

![image.png](./assets/1658024880604-image.png)

## 小结

* 在预测词元时，如果不是所有输入词元都是相关的，那么具有Bahdanau注意力的循环神经网络编码器-解码器会有选择地统计输入序列的不同部分。这是通过将上下文变量视为加性注意力池化的输出来实现的。
* 在循环神经网络编码器-解码器中，Bahdanau注意力将上一时间步的解码器隐状态视为查询，在所有时间步的编码器隐状态同时视为键和值。

## 正文

We studied the machine translation problem in [Section 9.7](https://d2l.ai/chapter_recurrent-modern/seq2seq.html#sec-seq2seq), where we designed an `encoder-decoder` architecture based on two RNNs for sequence to sequence learning. Specifically, the RNN encoder **transforms** a `variable-length sequence` **into** a `fixed-shape context variable`, then the RNN decoder generates the output (target) sequence token by token based on the generated tokens and the context variable. **However**, even though not all the input (source) tokens are useful for decoding a certain token, the **same context variable** that encodes the entire input sequence is still used at each decoding step. (我们需要改变环境变量)

In a separate but related challenge of handwriting generation for a given text sequence, Graves designed a differentiable attention model to align text characters with the much longer pen trace, where the alignment moves only in one direction [[Graves, 2013]](https://d2l.ai/chapter_references/zreferences.html#graves-2013). Inspired by the idea of learning to align, Bahdanau et al. proposed a `differentiable attention model` without the severe unidirectional alignment limitation [[Bahdanau et al., 2014]](https://d2l.ai/chapter_references/zreferences.html#bahdanau-cho-bengio-2014).

- When predicting a token, if not all the input tokens are relevant, the model aligns (or attends) only to **parts of the input sequence that are relevant to the current prediction**. This is achieved by treating the context variable as an output of attention pooling. (即通过 attention 机制，选择部分有用的输入序列)

## 4.1. Model

When describing **Bahdanau attention** for the RNN encoder-decoder below, we will follow the same notation in [Section 9.7](https://d2l.ai/chapter_recurrent-modern/seq2seq.html#sec-seq2seq). The new attention-based model is the same as that in [Section 9.7](https://d2l.ai/chapter_recurrent-modern/seq2seq.html#sec-seq2seq) except that the context variable $c$ in [(9.7.3)]() is replaced by $c_t^′$ at any decoding time step $t^′$. Suppose that there are $T$ tokens in the input sequence, the context variable at the decoding time step $t^′$ is the output of attention pooling:

$$
c_t^′=\sum_{t=1}^T\alpha(s_{t^′−1},h_t)h_t,

$$

where the **decoder** hidden state $s_{t^′−1}$ at time step $t^′−1$ is the `query`, and the **encoder** hidden states $h_t$ are both the `keys` and `values`, and the attention weight $\alpha$ is computed as in [(3.2)](./3_Attention_Scoring_Functions.md) using the **additive attention scoring function** defined by [(3.3)]().

Slightly different from the vanilla RNN encoder-decoder architecture in [Fig. 9.7.2](), the same architecture with **Bahdanau attention** is depicted in [Fig. 4.1]().

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/seq2seq-attention-details.svg" width = "75%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig 4.1 Layers in an RNN encoder-decoder model with Bahdanau attention.
  	</div>
</center>

## 4.2. Defining the Decoder with Attention

To implement the RNN encoder-decoder with Bahdanau attention, we **only need to redefine** the `decoder`. To visualize the learned attention weights more conveniently, the following `AttentionDecoder` class defines the base interface for decoders with attention mechanisms.

Now let us implement the RNN `decoder` with Bahdanau attention in the following `Seq2SeqAttentionDecoder` class.

The state of the decoder is initialized with

* (i) the $\color{blue}encoder$ $\colorbox{white}{{\color{magenta}\text{final-layer}}}$ hidden states at $\colorbox{white}{{\color{green}\text{all the time steps}}}$ $\colorbox{black}{{\color{yellow}\text{as}}}$ `keys` and `values` of the $\textbf{attention}$;

- (ii) the $\color{blue}encoder$ $\colorbox{white}{{\color{magenta}\text{all-layer}}}$ hidden state at $\colorbox{white}{{\color{green}\text{the final time step}}}$ $\colorbox{black}{{\color{yellow}\text{to initialize}}}$ the hidden state of the $\color{red}decoder$;
- (iii) the $\color{blue}encoder$ **valid length** (to exclude the padding tokens in attention pooling).

At each decoding time step,

- the $\color{red}decoder$ $\colorbox{white}{{\color{magenta}\text{final-layer}}}$ hidden state at the $\colorbox{white}{{\color{green}\text{previous time step}}}$ $\colorbox{black}{{\color{yellow}\text{is used as}}}$ the `query` of the $\textbf{attention}$.

As a result, both the attention output and the input embedding are concatenated as the input of the RNN decoder.

## 4.3. Training

Similar to [Section 9.7.4](https://d2l.ai/chapter_recurrent-modern/seq2seq.html#sec-seq2seq-training), here we specify hyperparemeters, instantiate an encoder and a decoder with Bahdanau attention, and train this model for machine translation. Due to the newly added attention mechanism, this training is much slower than that in [Section 9.7.4](https://d2l.ai/chapter_recurrent-modern/seq2seq.html#sec-seq2seq-training) without attention mechanisms.
