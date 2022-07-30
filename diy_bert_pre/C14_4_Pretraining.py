import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from C14_3_Dataset import *
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = load_data_ptb(batch_size, max_window_size, num_noise_words)


embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype}')

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    """
    返回center与噪声的向量内积
    :param center: 中心词, size = (batch, 1)
    :param contexts_and_negatives: 背景+负样本, size = (batch, maxlen)
    :param embed_v: 中心词的向量形式，size = (batch, 1, v)
    :param embed_u: 背景+噪声的向量形式, size = (batch, maxlen, v)
    :return: embed_v @ embed_u
    """
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1)) # 矩阵乘法, (b,1,v)*(b,v,maxlen)
    return pred

class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy Loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none"
        ) # with logits 表示输出自动加了一个softmax
        return out.mean(dim=1)

def sigmd(x):
    return -math.log(1 / (1+math.exp(-x)))

def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    """

    :param net:  Sequential(nn.Embedding (for centers), nn.Embedding (for context))
    :param data_iter: [[center, context_negative, mask, label], ...]
    :param lr: learning rate
    :param num_epochs:
    :param device:
    :return:
    """
    def init_weights(m):
        if (type(m)==nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # sum of normalize losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch
            ]
            # 计算center与context_negative的相关度
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                 / mask.sum(axis=1) * mask.shape[1] # 除以 mask 的均值
                 )
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i+1)%(num_batches // 5)==0 or i==num_batches-1:
                animator.add(epoch+(i+1) / num_batches,
                             (metric[0] / metric[1], ))
        print(f'loss {metric[0]/metric[1]: .3f}, '
              f'{metric[1] / timer.stop(): .1f} tokens/sec on {str(device)}')

def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W*W, axis=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype("int32")
    for i in topk[1:]: # remove the input words
        print(f'cosine sim={float(cos[i]): .3f}: {vocab.to_tokens(i)}')



if __name__ == "__main__":
    loss = SigmoidBCELoss()
    # pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]]*2) ## 数组里面内容翻倍
    # lable = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    # mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    # loss(pred, lable, mask) * mask.shape[1] / mask.sum(axis=1)

    embed_size = 100
    net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size))

    lr, num_epochs = 0.002, 5
    # train(net, data_iter, lr, num_epochs)

    get_similar_tokens('chip', 3, net[0])