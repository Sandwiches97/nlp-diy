import math
import torch
from d2l_en.pytorch.d2l import torch as d2l

def adgrad_2d(x1, x2, s1, s2):
    """ y = 0.1*(x1)^2 + 2*(x2)^2"""
    eps = 1e-6
    g1, g2 = 0.2*x1, 4*x2
    s1 += g1**2
    s2 += g2**2
    x1 -= eta/math.sqrt(s1+eps)
    x2 -= eta/math.sqrt(s2+eps)
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1*x1**2 + 2*x2**2

def init_adagrad_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()

def main():
    pass


if __name__=="__main__":
    eta=0.4
    d2l.show_trace_2d(f_2d, d2l.train_2d(adgrad_2d))