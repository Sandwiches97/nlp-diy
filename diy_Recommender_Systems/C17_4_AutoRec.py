import torch
from d2l_en.pytorch.d2l import torch as d2l
from torch import autograd, nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from utilts import try_gpu
from C17_2_ml_dataset import read_data_ml100k, split_data_ml100k, load_data_ml100k
from C17_3_matrix_factorization import train_recsys_rating

class AutoRec(nn.Module):
    def __init__(self, input_size, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(input_size, num_hidden, bias=True)
        self.decoder = nn.Linear(num_hidden, num_users, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input:torch.Tensor):
        hidden = self.dropout(F.sigmoid(self.encoder(input.to(dtype=torch.float32))))
        pred = self.decoder(hidden)
        if input.requires_grad:
            return pred*torch.sign(input)
        else:
            return pred

def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = values.to(devices)
        scores.extend([network(i) for i in feat])
    recons = torch.Tensor([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = torch.sqrt(
        torch.sum(
            torch.square(test_data-torch.sign(test_data)*recons)
        )/torch.sum(torch.sign(test_data))
    )
    return float(rmse)

if __name__=="__main__":
    device = try_gpu()
    df, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(df, num_users, num_items)
    _, _, _, train_inter_mat = load_data_ml100k(train_data, num_users,
                                                    num_items)
    _, _, _, test_inter_mat = load_data_ml100k(test_data, num_users,
                                                   num_items)
    train_iter = DataLoader(train_inter_mat, shuffle=True, drop_last=True, batch_size=256)
    test_iter = DataLoader(test_inter_mat, shuffle=False, drop_last=False, batch_size=1024)

    net = AutoRec(num_users, 500, num_users)
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0., std=0.01)
    net.apply(init_weights)
    net.to(device)
    lr, num_epochs, wd = 0.002, 20, 1e-5
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    train_recsys_rating(net, train_iter, test_iter, loss, optimizer, num_epochs, device, evaluator, inter_mat=test_inter_mat)

