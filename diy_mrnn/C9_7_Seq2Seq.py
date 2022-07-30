import collections
import math
import torch
from torch import nn
from typing import List
from diy_RNN.C8_2_Text_Preprocessing import Vocab
from d2l_en.pytorch.d2l import torch as d2l
from diy_mrnn.C9_6_Encoder_Decoder import EncoderDecoder, Encoder, Decoder
from diy_torch.C3_linear_networks.C3_1_LR import Timer
from diy_torch.C3_linear_networks.C3_6_softmax_scratch import Accumulator, Animator

#@save
class Seq2SeqEncoder(Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 num_hiddens: int,
                 num_layers: int,
                 dropout=0., **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros
        output, state = self.rnn(X)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state

class Seq2SeqDecoder(Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0., **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state

#@save
def sequence_mask(X: torch.tensor, valid_len: torch.tensor, value=0):
    """ Mask irrelevant entries in sequences.

    :param X: tensor, (batch, sequences, num_hidden)
    :param valid_len: tensor, (batch, int)
    :param value:
    :return: tensor, (batch, sequences, num_hidden)
    """
    maxlen = X.size(1) # 返回：当前tesor 列的维f_m度
    # torch.arange() 相当于 range函数
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value # 大于valid_len的token全部置零
    return X

#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none' # CrossEntropy().forward()时，直接返回，而不是mean或sum
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label) # 调用父类，nn.CrossEntropyLoss.forward() 方法
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab: Vocab, device):
    """ Train a model for seq2seq

    :param net:
    :param data_iter:
    :param lr:
    :param num_epochs:
    :param tgt_vocab:
    :param device:
    :return:
    """
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 将 <bos> 与 (label Y) concatenate起来，并去掉 <ens>
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学，teacher forcing
            # 1. forward
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # 2. 计算loss
            l = loss(Y_hat, Y, Y_valid_len)
            # 3. backward
            optimizer.zero_grad()
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            # 4. 更新参数
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab: Vocab, num_steps,
                    device, save_attention_weights=False):
    """ Predict for sequence to sequence.

    :param net:
    :param src_sentence:
    :param src_vocab:
    :param tgt_vocab:
    :param num_steps: 预测句子的最大长度
    :param device:
    :param save_attention_weights:
    :return:
    """
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[
                     src_sentence.lower().split(' ')] + [src_vocab['<eos>']
                                                         ]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>']) # 少补多砍
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0
    )
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0
    )
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps): # 预测句子的最大长度 T'
        Y, dec_state = net.decoder(dec_X, dec_state) # Y: (, , vocab size)
        dec_X = Y.argmax(dim=2) # 贪心：使用具有预测最高的词元，作为解码器在下一时间步的输入
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def main():
    # encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
    #                          num_layers=2)
    # encoder.eval()
    # X = torch.zeros((4, 7), dtype=torch.long) #  = 4 个 batchsize； 7个单词
    # # output, state = encoder(X)
    #
    # decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
    #                          num_layers=2)
    # decoder.eval()
    # state = decoder.init_state(encoder(X))
    # output, state = decoder(X, state)
    # print(output)
    #
    # X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # sequence_mask(X, torch.tensor([1, 2]))

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device
        )
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

if __name__ == "__main__":
    loss = MaskedSoftmaxCELoss()
    loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
         torch.tensor([4, 2, 0]))

    main()