import math
import time
import numpy as np
import torch
from d2l_en.pytorch.d2l import torch as d2l

class Timer:
    """Record multiple running times, 记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        """启动计时器"""
        self.tik = time.time()
    def stop(self):
        """停止计时器，并将时间记录在List中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)
    def sum(self):
        """返回时间总和"""
        return sum(self.times)
    def cumsum(self):
        """返回累计时间，计算前缀和"""
        return np.array(self.times).cumsum().tolist()

def main():
    n = 10000
    a, b = torch.ones(n), torch.ones(n)
    c = torch.zeros(n)
    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f'{timer.stop():.5f} sec')

    timer.start()
    m = map(lambda x,y: x+y, a, b)
    print(f'{timer.stop():.5f} sec')

    timer.start()
    d = a+b
    print(f'{timer.stop():.5f} sec')

if __name__ == "__main__":
    main()