import torch
from torch import nn

def try_gpu(i=0):
    """如果存在，则返回 gpu(i), 否则返回 cpu() """
    if torch.cuda.device_count() >= i+1:
        return  torch.device(f"cuda:{i}")
    return torch.device("cpu")

def try_all_gpus():
    """ 返回所有可用GPU，如果没有GPU，则返回[cpu(), ]"""
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]

def main():
    torch.device('cpu')