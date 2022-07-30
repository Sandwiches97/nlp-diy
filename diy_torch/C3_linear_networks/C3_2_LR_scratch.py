import random
import torch
from d2l_en.pytorch.d2l import torch as d2l

def syntetic_data(w, b, num_examples):
    """ 生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机的，没有特定顺序
    random.shuffle(indices) # 打乱索引
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        # yield 类似于return，只有调用next函数才会继续进行, 否则将暂停在原地
        yield features[batch_indices], labels[batch_indices]

def linearRegModer(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def sgd(params:list, lr, batch_size):
    """mini batch SGD"""
    with torch.no_grad(): # 下面模块的计算，不会更新梯度
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()     # 上述语句用来“清除”x的梯度值，也就是重新赋值为0。
                                            #只有当x被求过一次梯度的时候，这个函数才能使用，否则会报错。
def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = syntetic_data(true_w, true_b, 1000)
    print('features:', features[0], '\nlabel:', labels[0])

    batch_size = 10
    # for X, y in data_iter(batch_size, features, labels):
    #     print(X, '\n', y)
    #     break

    # 初始化参数
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 超参数
    lr = 0.03
    num_epochs = 3
    net = linearRegModer
    loss = squared_loss


    # 训练 loop
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            y_hat = net(X, w, b)                   # 获得模型预测值
            l = loss(y_hat, y)                        # 计算损失
            l.sum().backward()                      # 后向传播计算梯度，sum()表示对batch求和
            sgd([w, b], lr, batch_size)         # 更新参数 （细节，传入列表引用）

        with torch.no_grad(): # 训练完一个 epoch 后
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')





if __name__ == "__main__":
    main()