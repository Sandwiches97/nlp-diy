# 5. Multi-Head Attention

* 多头注意力融合了来自于多个注意力 (多组Query/Key/Value权重矩阵，这些权重矩阵集合中的每一个都是随机初始化的) 汇聚的不同知识，这些知识的不同来源于**相同**的查询、键和值的**不同的子空间`W`表示**。多个head学习到的Attention侧重点可能略有不同，这样给了模型更大的容量。
* 基于适当的张量操作，可以实现多头注意力的并行计算。（`W`被切分成 -> `W/num_heads`）

In practice, given the same set of `queries`, `keys`, and `values` we may want our model to combine knowledge from **different behaviors** of the **same attention mechanism**, such as capturing dependencies of various ranges (e.g., shorter-range vs. longer-range) within a sequence. Thus, it may be beneficial to allow our attention mechanism to jointly use different representation subspaces of `queries`, `keys`, and `values`.

To this end, instead of performing a single attention pooling, queries, keys, and values can be transformed with $h$ independently learned $linear\ projections$ (namely, $FC$ in fig 5.1). Then these $h$ projected `queries`, `keys`, and `values` are fed into attention pooling in parallel. In the end, $h$ attention pooling outputs are concatenated and transformed with another learned $linear\ projection$ to produce the final output. This design is called  **multi-head attention** , where each of the $h$ attention pooling outputs is a **head** [[Vaswani et al., 2017]](https://d2l.ai/chapter_references/zreferences.html#vaswani-shazeer-parmar-ea-2017). Using fully-connected layers to perform learnable linear transformations, [Fig. 10.5.1](https://d2l.ai/chapter_attention-mechanisms/multihead-attention.html#fig-multi-head-attention) describes multi-head attention.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/multi-head-attention.svg" width = "70%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig 5.1 Multi-head attention, where multiple heads are concatenated then linearly transformed.
  	</div>
</center>

## 5.1. Model

Before providing the implementation of **multi-head attention**, let us formalize this model mathematically. Given a `query` $q\in \mathbb{R}^{d_q}$, a `key` $k \in \mathbb{R}^{d_k}$ and a `value` $v\in \mathbb{R}^{d_v}$, each attention head $h_i (i=1,…,h)$ is computed as

$$
h_i=f(W_i^{(q)}q,W_i^{(k)}k,W_i^{(v)}v)\in \mathbb{R}^{p_v}, \tag{5.1}

$$

where learnable parameters $W_i^{(q)}\in\mathbb{R}^{p_q×d_q}$, $W_i^{(k)}\in\mathbb{R}^{p_k×d_k}$ and $W_i^{(v)}\in\mathbb{R}^{p_v×d_v}$, and $f$ is attention pooling, such as **additive attention** and **scaled dot-product attention** in [Section 3](./3_Attention_Scoring_Functions.md). The **multi-head attention output** is **another linear transformation** ($FC$) via learnable parameters $W_o\in \mathbb{R}^{po×hpv}$ of the concatenation of h heads:

$$
W_o\cdot [h_1,...,h_h]^T\in\mathbb{R}^{po}.

$$

Based on this design, each head may attend to different parts of the input. More sophisticated functions than the simple weighted average can be expressed.

```python
import  math
import torch
from   torch import  nn
from d2l_en.pytorch.d2l import torch as d2l
from C10_3_Attention_Scoring_Function import DotProductAttention
```

## 5.2. Implementation

In our implementation, we choose the **scaled dot-product attention** for each head of the multi-head attention. To avoid significant growth of computational cost and parameterization cost, we set $p_q=p_k=p_v=p_o/h$. Note that $h$ heads can be computed in parallel if we set the number of outputs of linear transformations for the `query`, `key`, and `value` to $p_qh=p_kh=p_vh=p_o$. In the following implementation, $p_o$ is specified via the argument `num_hiddens`.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """ 

        :param queries: (`batch_size`, no. of queries, `num_hiddens`)
        :param keys:     (`batch_size`, no. of key-value pairs, `num_hiddens`)
        :param values:  (`batch_size`, no. of  key-value pairs, `num_hiddens`)
        :param valid_lens:  (`batch_size`,) or (`batch_size`, no. of queries)
        :return: (`batch_size`, no. of queries, `num_hiddens`)
        """
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`,  no. of queries or key-value pairs,  `num_hiddens` / `num_heads`)

        if valid_lens is not  None:
            # on axis 0, copy the first item (scalar or vector) for " num_head " times,
            # then copy the next item, ans so on.
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,  `num_hiddens` / `num_heads`)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

To allow for parallel computation of multiple heads, the above `MultiHeadAttention` class uses two transposition functions as defined below. Specifically, the `transpose_output` function reverses the operation of the `transpose_qkv` function.

```python
def transpose_qkv(X, num_heads):
    """ Transposition for parallel computation of multiple attention heads.

    :param X:   (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    :param num_heads:  (`batch_size` * `num_heads`, no. of queries or key-value pairs,  `num_hiddens` / `num_heads`)
    :return: (`batch_size` * `num_heads`, no. of queries or key-value pairs,  `num_hiddens` / `num_heads`)
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,  `num_hiddens` / `num_heads`)

    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """ Reverse the operation of `transpose_qkv`.

    :param X:   (`batch_size` * `num_heads`, no. of queries,  `num_hiddens` / `num_heads`)
    :param num_heads:
    :return:  (`batch_size`, no. of queries, `num_hiddens`).
    """
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

Let us test our implemented `MultiHeadAttention` class using a toy example where keys and values are the same. As a result, the shape of the multi-head attention output is `(batch_size, num_queries, num_hiddens)`.

```python
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()
```

```python
batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
attention(X, Y, Y, valid_lens).shape
```
