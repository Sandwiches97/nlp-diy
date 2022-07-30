import torch
from torch import nn
from torch.nn import functional as F
from d2l_en.pytorch.d2l import torch as d2l
from C8_3_language_models import load_data_time_machine
from C8_5_RNN_scratch import train_ch8, predict_ch8

class RNNModel(nn.Module):
    """ The RNN model"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens*2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU 以 tensor 作为 hidden state
            return torch.zeros(
                (
                    self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens
                ),
                device=device
            )
        else:
            # nn.LSTM 以 tuple 作为 hidden state
            return (torch.zeros(
                (
                    self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens
                ),
                device=device
            ))

def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    num_hiddens = 26
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    state = torch.zeros((1, batch_size, num_hiddens))

    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state)

    device = d2l.try_gpu()
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)
    predict_ch8('time traveller', 10, net, vocab, device)

    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)

if __name__ == "__main__":
    main()