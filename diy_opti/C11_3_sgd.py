import numpy as np
import torch
from d2l_en.pytorch.d2l import torch as d2l



def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f"epoch 10, x: {x:f}")
    return results

def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

def train_2d(trainer, steps=20, f_grad=None):
    """ optimize a 2-dim objective function with a customized trainer

    :param trainer: 梯度下降法， 例如 gd(), gd_2d()
    :param steps: 迭代步数
    :param f_grad: 目标函数的梯度，用于对比
    :return:
    """
    # `s1` 与 `s2` 是稍后将使用的内部状态变量
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
    print(f"epoch {i+1}, x1: {float(x1):f}, x2: {float(x2): f}")
    return results

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)



def main():
    f = lambda x: x**2
    f_grad = lambda x: 2*x
    results = gd(0.2, f_grad)
    show_trace(results, f)

if __name__=="__main__":
    eta = 0.1
    main()