import torch


if __name__ == "__main__":
    x = torch.arange(4.0)
    x.requires_grad_(True)
    y = 2*torch.dot(x, x)
    print(y)
    y.backward()
    print(x, x.grad)