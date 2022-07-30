import torch



if __name__ == "__main__":
    X = 12
    before = id(X)
    X += 13
    print(id(X)==before)

    X_t = torch.tensor([1, 3])
    Y_t = torch.tensor([2, 4])
    X_t_before = id(X_t)
    X_t += Y_t
    print(id(X_t)==X_t_before)