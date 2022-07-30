import  math
import torch
from   torch import  nn
from d2l_en.pytorch.d2l import torch as d2l
from diy_Attention.C10_3_Attention_Scoring_Function import DotProductAttention


def transpose_qkv(X, num_heads):
    """ Transposition for parallel computation of multiple attention heads， 加速运算

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
            # then copy the next item, ans so on. 重复 头 次
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,  `num_hiddens` / `num_heads`)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)