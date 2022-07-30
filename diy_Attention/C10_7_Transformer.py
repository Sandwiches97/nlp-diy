import math
import pandas as pd
import torch
from torch import nn
from d2l_en.pytorch.d2l import torch as d2l
from diy_Attention.c10_5_Multi_Head_Attention import MultiHeadAttention
from diy_Attention.C10_6_Self_Attention_Positional_Encoding import PositionalEncoding
from diy_mrnn.C9_7_Seq2Seq import train_seq2seq, predict_seq2seq

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """Residual connection followed by layer normalization"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(self, key_size, query_size, value_size, num_hiddens, normalized_shape,
                 ffn_num_input, ffn_num_hiddens,  num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention( key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(normalized_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens) # 因为我们需要与输入相同形状的输出，所以输出设置为了num_hiddens
        self.addnorm2 = AddNorm(normalized_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.attention(X, X, X, valid_lens)
        Y = self.addnorm1(X, Y)
        output = self.addnorm2(Y, self.ffn(Y))
        return output

class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding =PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock( key_size, query_size, value_size, num_hiddens, norm_shape,
                              ffn_num_input,  ffn_num_hiddens, num_heads, dropout, use_bias)
            )

    def forward(self, X, valid_lens, *args):
        """ Since positional encoding values are between -1 and 1, the embedding
         values are multiplied by the square root of the embedding dimension
         to rescale before they are summed up

        :param X:
        :param valid_lens:
        :param args:
        :return:
        """
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, block in enumerate(self.blks):
            X = block(X, valid_lens)
            self.attention_weights[i] = block.attention.attention.attention_weights
        return X

class DecoderBlock(nn.Module):
    """ The `i`-th block in the decoder, 这里的 i 是个标识符"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i

        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        """
        - In the encoder-decoder attention,
                queries are from the outputs of the previous decoder layer,
                and the keys and values are from the transformer encoder outputs.
        - In the decoder self-attention,
                queries, keys, and values are all from the the outputs of the previous decoder layer

        During 训练阶段, all the tokens of any output sequence are processed at 同一时刻处理,
        so `state[2][self.i]` is `None` as initialized.
        during 预测阶段, 输出序列时通过次元一个接着一个解码的，
        因此 `state[2][self.i]` 包含着until 当前时间步第 i 个块解码的输出表示

        :param X: X.shape = (batch size, 句子长度, attention矩阵的维度=768)
        :param state: [the output of EncoderBlock, valid_lens, [None]]
        :return:
        """
        enc_outputs, enc_valid_lens = state[0], state[1] # 存来自Transformer encoder的结果
        if (state[2][self.i] is None):
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens就是掩码 ，
            # Shape of `dec_valid_lens`: (`batch_size`, `num_steps`),
            # where every row is [1, 2, ..., `num_steps`]  每行都是一个从1增加到`num_steps`的数组
            dec_valid_lens = torch.arange(
                1, num_steps+1, device=X.device
            ).repeat(batch_size, 1) # 对批次中的每个句子都重复该掩码操作
        else:
            dec_valid_lens = None

        # self-attention, dec_valid_lens会使得后面的输出都变成
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        #  Encoder-decoder attention. Shape of `enc_outputs`:  (`batch_size`, `num_steps`, `num_hiddens`)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class AttentionDecoder(d2l.Decoder):
    """The base attention-based decoder interface.

    Defined in :numref:`sec_seq2seq_attention`"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i)
            )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        """

        :param X:
        :param state: 这是encoder的
        :return:
        """
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Encoder self-attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

def train():
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads,
                                                                                 -1, num_steps))
def showDecoder():
    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()

    decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
    decoder_blk.eval()
    state = [encoder_blk(X, valid_lens), valid_lens, [None]]
    print(decoder_blk(X, state)[0].shape)
def showDiff_LN_BN():
    ln = nn.LayerNorm(2)
    bn = nn.BatchNorm1d(2)
    X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    # compute mean and variance from 'X' in the training mode
    print(f"X: {X}")
    print(f'layer norm: {ln(X)}, \nbatch norm: {bn(X)}')
def showEncoder():
    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    print(f"the shape of X is {X.shape }, and valid lens are {valid_lens}")
    print(f"the output of Encoder is {encoder_blk(X, valid_lens).shape}")
def showTransEncoder():
    encoder = TransformerEncoder(
        200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    encoder.eval()
    valid_lens = torch.tensor([3, 2])
    print(f"两个句子输入到两层的Transformer encoder中，得到的结果为：\n {encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape}")
if __name__ == "__main__":
    # ffn = PositionWiseFFN(4, 4, 8)
    # ffn.eval()
    # print((f"the output of torch.ones((2, 3, 4))  after the PositionWise FFN operation is: \n{ffn(torch.ones((2, 3, 4)))[0]}"))

    train()