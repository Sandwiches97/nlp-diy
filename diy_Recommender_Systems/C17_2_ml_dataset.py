import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from d2l_en.pytorch.d2l import torch as d2l
from typing import Tuple
from pandas import DataFrame

d2l.DATA_HUB['ml-100k'] = (
    'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')


def read_data_ml100k() -> Tuple[DataFrame, int, int]:
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names,
                       engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items

def statistic_dataset():
    data, num_users, num_items = read_data_ml100k()
    sparsity = 1 - len(data) / (num_users * num_items)
    print(f'number of users: {num_users}, number of items: {num_items}')
    print(f'matrix sparsity: {sparsity:f}')
    print(data.head(5))

    d2l.plt.hist(data['rating'], bins=5, ec='black')
    d2l.plt.xlabel('Rating')
    d2l.plt.ylabel('Count')
    d2l.plt.title('Distribution of Ratings in MovieLens 100K')
    d2l.plt.show()

def split_data_ml100k(data, num_users, num_items, split_mode="random", test_ratio=0.1):
    """ Split the dataset in random mode or seq-aware mode. """
    if split_mode=="seq-aware":
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users+1):
            train_list.extend(sorted(train_items[u], key=lambda x: x[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else: # the 90% of the data as training samples and the rest 10% as test samples by default
        mask = [True if x==1 else False for x in np.random.uniform(0, 1, (len(data))) < 1-test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data

def load_data_ml100k(data: DataFrame, num_users: int, num_items: int, feedback="explicit") -> Tuple:
    """

    :param data:
    :param num_users:
    :param num_items:
    :param feedback:
    :return:  rating matrix (the user-item interaction matrix)
        - a matrix shape = (num_users, num_items), and value = score
        - or a dict
    """
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback=="explicit" else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2]-1)
        score = int(line[3]) if feedback=="explicit" else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback=="implicit":
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter

class Data(Dataset):
    def __init__(self, train_u, train_i, train_r ):
        self.data = [(train_u[i], train_i[i], train_r[i]) for i in range(len(train_u))]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def split_and_load_ml100k(split_mode="seq-aware", feedback="explicit",
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(test_data, num_users, num_items, feedback)

    # tmp_u, tmp_i = np.zeros((len(train_u), num_users), dtype=int), np.zeros((len(train_u), num_items), dtype=int)
    # tmp_u[np.arange(len(train_u)), train_u] = 1
    # tmp_i[np.arange(len(train_u)), train_i] = 1
    train_u, train_i, train_r = np.array(train_u), np.array(train_i), np.array(train_r)

    # tmp_u, tmp_i = np.zeros((len(test_u), num_users), dtype=int), np.zeros((len(test_u), num_items), dtype=int)
    # tmp_u[np.arange(len(test_u)), test_u] = 1
    # tmp_i[np.arange(len(test_u)), test_i] = 1
    test_u, test_i, test_r = np.array(test_u), np.array(test_i), np.array(test_r)

    train_ds = Data(train_u, train_i, train_r)
    test_ds = Data(test_u, test_i, test_r)
    train_iter = DataLoader(train_ds, shuffle=True, drop_last=False, batch_size=batch_size)
    test_iter = DataLoader(test_ds, shuffle=False, drop_last=False, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter




def main():

    split_and_load_ml100k()



if __name__=="__main__":

    main()