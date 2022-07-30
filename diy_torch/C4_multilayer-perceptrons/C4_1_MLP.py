import torch
from d2l_en.pytorch.d2l import torch as d2l

def main():
    pass
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.tanh(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
    d2l.plt.show()


if __name__ == "__main__":
    main()