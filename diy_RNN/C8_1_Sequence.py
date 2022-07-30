import torch
from torch import nn
from d2l_en.pytorch.d2l import torch as d2l

# Function for initializing the weights of the network
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# A simple MLP
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')



def main():
    T = 1000  # Generate a total of 1000 points
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

    tau = 4
    ## 这样设计实际上太过于简单，
    # 训练集 {(x1, x2, x3,x4, y=x5), (x2, x3,x4, x5, y=x6), ...}，
    # 测试集 {(x601, x602, x603,x604, y=x605), (x602, x603,x604, x605, y=x606), ...}
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]
    labels = x[tau:].reshape((-1, 1))


    batch_size, n_train = 16, 600
    # Only the first `n_train` examples are used for training
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                                batch_size, is_train=True)

    # Note: `MSELoss` computes squared error without the 1/2 factor
    loss = nn.MSELoss(reduction='none')

    net = get_net()
    train(net, train_iter, loss, 5, 0.01)

    onestep_preds = net(features)
    d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
             'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
    d2l.plt.show()

    multistep_preds = torch.zeros(T)
    multistep_preds[: n_train + tau] = x[: n_train + tau]
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(
            multistep_preds[i - tau:i].reshape((1, -1)))
    d2l.plot([time, time[tau:], time[n_train + tau:]],
             [x.detach().numpy(), onestep_preds.detach().numpy(),
              multistep_preds[n_train + tau:].detach().numpy()], 'time',
             'x', legend=['data', '1-step preds', 'multistep preds'],
             xlim=[1, 1000], figsize=(6, 3))

    max_steps = 64
    features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
    # Column `i` (`i` < `tau`) are observations from `x` for time steps from
    # `i + 1` to `i + T - tau - max_steps + 1`
    for i in range(tau):
        features[:, i] = x[i: i + T - tau - max_steps + 1]

    # Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
    # time steps from `i + 1` to `i + T - tau - max_steps + 1`
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau:i]).reshape(-1)

    steps = (1, 4, 16, 64)
    d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
             [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
             legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
             figsize=(6, 3))

if __name__ == "__main__":
    main()