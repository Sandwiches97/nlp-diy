import math
import torch
from torch import nn
from d2l_en.pytorch.d2l import torch as d2l
from diy_Attention.c10_5_Multi_Head_Attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough 'P', shape = (1, 句子最大长度（表示单词在句子中的位置）， 句子长度)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1
                ) / torch.pow(10000, torch.arange(
                        0, num_hiddens, 2, dtype=torch.float32)/num_hiddens
                    )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

def test_6_2():
    encoding_dim, num_steps = 32, 60 # 60个单词，32维度的词向量
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]
    d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
             figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
    d2l.plt.show()

if __name__ == "__main__":
    test_6_2()
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)

    attention.eval()

    batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    print(f"the output size of attention: {attention(X, X, X, valid_lens).shape}")