import torch
from torch import nn


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bais)

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

def my_init(m):
    if type(m) == nn.Linear:
        print("init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5



def main():
    pass
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))

    print(net[2].state_dict())
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)
    net.apply(init_normal)

    # 给共享层一个名称，以便可以引用它的参数
    shared = nn.Linear(8, 8)
    net = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(),
        shared, nn.ReLU(),
        shared, nn.ReLU(),
        nn.Linear(8, 1)
    )
    net(X)


if __name__ == "__main__":
    main()