import torch
from torch import autograd, nn
from torch import utils
from torch.utils.data import DataLoader
from utilts import try_gpu
from C17_2_ml_dataset import read_data_ml100k, split_data_ml100k, load_data_ml100k
from C17_3_matrix_factorization import train_recsys_rating


class NeuMF(nn.Module):
    def __init__(self, input_size, num_factors, num_users, num_items, nums_hiddens):
        super(NeuMF, self).__init__()
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)
        self.mlp = nn.Sequential()
        for num_hiddens in nums_hiddens:
            self.mlp.add(nn.Linear(input_size, num_hiddens,
                                  bias=True))
            self.mlp.add(nn.ReLU())
        self.prediction_layer = nn.Linear(nums_hiddens[-1], 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(torch.cat((p_mlp, q_mlp), dim=1))
        con_res = torch.cat((gmf, mlp), dim=1)
        return self.prediction_layer(con_res)

class PRDataset(torch.utils.data.Dataset):
    def __init__(self, users, items, candidates, num_items):
        super(PRDataset, self).__init__()
        self.users = users
        self.items = items
        self.cand = candidates
        self.all = set([i for i in range(num_items)])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg_items = list(self.all - set(self.cand[int(self.users[idx])]))
        indices = torch.randint(len(neg_items) - 1, (len(neg_items) - 1, ))
        return self.users[idx], self.items[idx], neg_items[indices]