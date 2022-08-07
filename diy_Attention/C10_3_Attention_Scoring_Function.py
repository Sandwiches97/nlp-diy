import math
import torch
from torch import nn
from d2l_en.pytorch.d2l import torch as d2l

def masked_softmax(X, valid_lens):
    """ 在最后一个轴上mask元素

    :param X: shape = (batch size, sentence number, tokens number)
    :param valid_lens: shape = (batch size, ) 对每个批次，都有设置一个有效长度
    :return:
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
# masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))


class AdditiveAttention(nn.Module):
    """Additive attention"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # After dimension expansion, 将queries与keys的维度错开
        # shape of "queries": ("batch_size", queries 的数量, 1, "num_hiddens") and
        # shape of "keys": ("batch_size", 1, key-value pairs 的数量 , "num_hiddens").
        # Sum them up with broadcasting
        features = torch.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last
        # one-dimensional entry from the shape.
        scores = self.W_v(features).squeeze(-1) # 去掉 shape=1 的 axis,
                                                                    # score.shape = (`batch_size`, no. of queries, no. of key-value pairs)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of `values`:
        # (`batch_size`, no. of key-value pairs, value dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """  注意这里维护了一个属性self.attention_weights: 它是一个矩阵，其中的元素表示 queries中的每个query与每个键值对的匹配分数
        (`batch_size`, no. of queries || pairs, `d`||`v`) ===>>>>   (`batch_size`, no. of queries,  no. of key-value pairs)
        其中，这里面的`d`是 （k,q的词向量维度/num_heads）， `v`是（value的长度/num_heads），在自注意力机制中，'d'='v'

        :param queries: Shape of `queries`: (`batch_size`, no. of queries, `d`)
        :param keys:    Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
        :param values:  Shape of `values`: (`batch_size`, no. of key-value pairs, `v`: value's length)
        :param valid_lens:  Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries 查询的个数)
        :return: attention_weights * values, shape = (`batch_size`, no. of key-value pairs, `v`)
        """
        d = queries.shape[-1] #  d_q = d_k = d_v
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == "__main__":
    queries, keys = torch.normal(0, 1, size=(2, 1, 20)), torch.ones((2, 10, 2))
    # The two value matrices in the `values` minibatch are identical
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                                  dropout=0.1)
    attention.eval() # 不使用BN层与Dropout，模型只验证不训练
    attention(queries, keys, values, valid_lens)

    queries = torch.normal(0, 1, (2, 1, 2))
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    attention(queries, keys, values, valid_lens)