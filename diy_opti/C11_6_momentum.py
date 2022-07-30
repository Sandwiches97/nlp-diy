import torch
from d2l_en.pytorch.d2l import torch as d2l
from C11_5_mini_batch_sgd import train_ch11

def init_momentum_states(feature_dim):
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams["momentum"] * v + p.grad
            p[:] -= hyperparams["lr"] * v
        p.grad.data.zero_()

def train_momentum(lr, momentum, num_epochs=2):
    train_ch11(sgd_momentum, init_momentum_states(feature_dim),
               hyperparams={"lr": lr, "momentum": momentum}, data_iter=data_iter,
               feature_dim=feature_dim, num_epochs=num_epochs)

def main():
    train_momentum(0.02, .05)

if __name__=="__main__":
    data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    main()