import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l_en.pytorch.d2l import torch as d2l
from diy_RNN.C8_3_language_models import load_data_time_machine
from diy_torch.C5_deep_learning_computation.C5_6_GPU import try_gpu
from diy_torch.C3_linear_networks.C3_1_LR import Timer
from diy_torch.C3_linear_networks.C3_2_LR_scratch import sgd
from diy_torch.C3_linear_networks.C3_6_softmax_scratch import Accumulator

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    """

    :param inputs:  shape = (nums_steps, batch_size, vocab_size)
    :param state:
    :param params:
    :return:
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs: # X.shape = (batch size, vocab_size)
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, )

class RNNModelScratch:
    """ A RNN model implemented from scratch. """
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state, *args, **kwargs):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

def predict_ch8(prefix, num_preds, net, vocab, device):
    """ 在 prefix 后面，生成新的字符

    :param prefix:  a string containing several characters
    :param num_preds:    单词长度（字符级别）
    :param net:
    :param vocab: 字符级别的 vocab，len(vocab) = 28
    :param device:
    :return:
    """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda : torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    for y in prefix[1:]:                        # warm up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):          # 预测 num_preds step
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ' '.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    """

    :param net: a class 引用传入
    :param theta: int or float
    :return:
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta/norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """ 训练网咯 一个迭代周期

    :param net:
    :param train_iter:
    :param loss:
    :param updater:
    :param device:
    :param use_random_iter:
    :return:
    """
    state, timer = None, Timer()
    metric = Accumulator(2) # 训练loss之和，词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用 random sampling 时 初始化 state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state 对于 nn.GRU 是个 tensor
                state.detach_()
            else:
                # state 对于 nn.LSTM 或对于我们 scratch模型 是个 tuple
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean 函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """ 训练 net

    :param net:
    :param train_iter:
    :param vocab:
    :param lr:
    :param num_epochs:
    :param device:
    :param use_random_iter:
    :return:
    """
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size=batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # training and predicting
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter
        )
        if (epoch+1)%10 == 0:
            print(predict("time traveller"))
            animator.add(epoch+1, [ppl])
        print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
        print(predict('time traveller'))
        print(predict('traveller'))


def main():
    X = torch.arange(10).reshape((2, 5))

    batch_size, nums_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, nums_steps)
    num_hiddens=  512
    net = RNNModelScratch(len(vocab), num_hiddens,
                          device=try_gpu(), get_params=get_params, init_state=init_rnn_state, forward_fn=rnn)
    state = net.begin_state(X.shape[0], device=try_gpu())
    Y, new_state = net(X.to(try_gpu()), state)
    print(Y.shape, len(new_state), new_state[0].shape)

    predict_ch8("time traveller ", 10, net, vocab, try_gpu())

    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, try_gpu())

if __name__ == "__main__":
    main()