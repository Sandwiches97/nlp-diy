import torch
from torch import nn
from d2l_en.pytorch.d2l import torch as d2l
from diy_Attention.C10_7_Transformer import EncoderBlock


#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """ Get tokens of the BERT input sequence and their segment IDs.

    :param tokens_a: 句子1
    :param tokens_b: 句子2
    :return:
        - tokens of the BERT input sequence
        - their corresponding segment IDs.
    """
    tokens = ["<cls>"] + tokens_a + ["<sep>"]
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ["<sep>"]
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class BERTEncoder(nn.Module):
    """BERT encoder. """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True
            ))
        # In BERT, positional embedding are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        """ Shape of 'X' remains unchanged in the following code snippet:
        (batch size, max sequence length, 'num_hiddens')
        :param tokens:
        :param segments:
        :param valid_lens:
        :return:
        """
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

#@save
class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        """

        :param X:
            the encoded result of BERTEncoder
        :param pred_positions:
            the token positions for prediction.
        :return:
            the prediction results at these positions. (batch size, 句子长度, vocab 长度)
        """
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx.type(torch.long), pred_positions.type(torch.long)] # 被masked掉的 (batchsize * num_pred_positions) 个单词
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

#@save
class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT."""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`=句子长度*词典)
        return self.output(X)

#@save
class BERTModel(nn.Module):
    """ The BERT model.
    return encoded_X, mlm_Y_hat, nsp_Y_hat
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


def main():

    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)

    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None)
    print(encoded_X.shape)

    mlm = MaskLM(vocab_size, num_hiddens)
    mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]]) # 待预测的位置
    mlm_Y_hat = mlm(encoded_X, mlm_positions)
    print(mlm_Y_hat.shape)
    mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
    print(mlm_l.shape)



    # PyTorch by default won't flatten the tensor as seen in mxnet where, if
    # flatten=True, all but the first axis of input data are collapsed together
    encoded_X = torch.flatten(encoded_X, start_dim=1) # （batch size, 句子长度*词典）
    # input_shape for NSP: (batch size, `num_hiddens`)
    nsp = NextSentencePred(encoded_X.shape[-1])
    nsp_Y_hat = nsp(encoded_X)
    print(nsp_Y_hat.shape)

if __name__ == "__main__":
    main()