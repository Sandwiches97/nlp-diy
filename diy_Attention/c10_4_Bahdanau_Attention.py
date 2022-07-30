import torch
from torch import nn
import numpy as np
from d2l_en.pytorch.d2l import torch as d2l
from C10_3_Attention_Scoring_Function import AdditiveAttention

class AttentionDecoder(d2l.Decoder):
    """ The base Attention-based decoder interface."""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property # 创建只读属性，@property装饰器会将方法转换为相同名称的只读属性,可以与所定义的属性配合使用，这样可以防止属性被修改。
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout
        )
        self.embbeding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout
        )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        """
        :param enc_outputs:
        :param enc_valid_lens:
        :param args:
        :return:
        """
        outputs, hidden_state = enc_outputs
        # Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        #  Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        """

        :param X:
        :param state:
        :return:
        """
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
        X = self.embbeding(X).permute(1, 0, 2
                                      )# Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        outputs, self._attention_weights = [], []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1
                                    ) # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens
            ) # Shape of `context`: (`batch_size`, 1, `num_hiddens`)

            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)),
                          dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), # reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
                                         hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        outputs = self.dense(torch.cat(outputs, dim=0))# shape of `outputs`:  (`num_steps`, `batch_size`, `vocab_size`)
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

if __name__ == "__main__":
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = d2l.Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
              f'bleu {d2l.bleu(translation, fra, k=2):.3f}')