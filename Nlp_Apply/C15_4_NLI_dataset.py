import os
import re
import torch
from torch import nn
from torch.utils import data
from typing import Tuple
from d2l_en.pytorch.d2l import torch as d2l
from diy_RNN.C8_2_Text_Preprocessing import tokenize, Vocab

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')


def read_snli(data_dir:str, is_train:bool)->Tuple[list, list, list]:
    """ Read the SNLI dataset into premises, hypotheses, and labels.

    :param data_dir:
    :param is_train:
    :return: a Tuple[list, list, list] =  (premises, hypotheses, labels)
    """
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        # 用一个空格替换多个空格, \s：正则表达式匹配空白，tab键
        s = re.sub("\\s{2,}", " ", s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

class SNLIDataset(torch.utils.data.Dataset):
    """
    a customized dataset to load the SNLI dataset.
    """
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = tokenize(dataset[0])
        all_hypotheses_tokens = tokenize(dataset[1])
        if vocab is None:
            self.vocab = Vocab(all_premise_tokens + all_hypotheses_tokens,
                               min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypotheses_tokens)
        self.labels = torch.tensor(dataset[2])
        print("read" + str(len(self.premises)) + "examples")

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>']
        ) for line in lines])

    def __getitem__(self, item):
        return (self.premises[item], self.hypotheses[item], self.labels[item])

    def __len__(self): return len(self.premises)

def load_data_snli(batch_size, num_steps=50):
    """
    下载SNLI数据集，并返回数据集迭代器和vocab
    :param batch_size:
    :param num_steps:
    :return:
    """
    num_workers = d2l.get_dataloader_workers()
    data_dir = "E:\\FangC\\SourceCode\\pytorch\\NLP_diy\\data\\snli_1.0"
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, vocab=train_set.vocab)

if __name__ == "__main__":

    data_dir = "E:\\FangC\\SourceCode\\pytorch\\NLP_diy\\data\\snli_1.0"
    train_data = read_snli(data_dir, True)
    for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
        print(f'premise: {x0} \nhypothesis: {x1} \nlabel: {y}')

    test_data = read_snli(data_dir, is_train=False)
    for data in [train_data, test_data]:
        print([[row for row in data[2]].count(i) for i in range(3)])