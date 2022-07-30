import json
import multiprocessing
import os
import torch
from torch import nn
from torch.utils import data
from d2l_en.pytorch.d2l import torch as d2l
from diy_RNN.C8_2_Text_Preprocessing import Vocab, tokenize
from diy_bert_pre.C14_8_BERT import BERTModel, get_tokens_and_segments
from typing import Tuple, List
from C15_4_NLI_dataset import read_snli


d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')

def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens, num_heads,
                          num_layers, dropout, max_len, devices)->Tuple[BERTModel, Vocab]:
    """ 读取Bert的vocab、以及加载Bert模型的预训练parameter

    :param pretrained_model:
    :param num_hiddens:
    :param ffn_num_hiddens:
    :param num_heads:
    :param num_layers:
    :param dropout:
    :param max_len:
    :param devices:
    :return: a tuple = (Bert model, vocab)
    """
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, "vocab.json")))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    bert = BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                     ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens, num_heads=4,
                     num_layers=2, dropout=0.2, max_len=max_len,
                     key_size=256, query_size=256, value_size=256,
                     hid_in_features=256, mlm_in_features=256, nsp_in_features=256)
    # Load pretrained BERT parameters
    bert.load_state_dict(torch.load(os.path.join(data_dir, 'pretrained.params')))
    return bert, vocab

class SNLIBERTDatset(torch.utils.data.Dataset):
    def __init__(self, dataset:Tuple[list, list, list], max_len, vocab=None):
        """

        :param dataset:   a Tuple[list, list, list] =  (premises, hypotheses, labels)
        :param max_len:
        :param vocab:
        """
        # zip() 将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组。
        # 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，
        # 利用 zip(*) 即 * 号操作符（逆运算），可以将元组解压为列表。
        # 注意，两者返回的都是一个对象
        all_premise_hypothesis_tokens = [
            [p_tokens, h_tokens] for p_tokens, h_tokens in zip(*[
                tokenize([s.lower() for s in sentences]) for sentences in dataset[:2]
            ])
        ] # List[List[list, list], List[list, list], ...] = [[p_tokens, h_tokens], [p_tokens, h_tokens],..,]
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments, self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print("read " + str(len(self.all_token_ids)) + " examples")

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4) # Use 4 worker processses (多进程)
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens) # 就是一个循环
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out
        ]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_len = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_len))

    def _mp_worker(self, premise_hypothesis_tokens: List[list, list]):
        """ 将 p 与 h合并成一个 bert pair input

        :param premise_hypothesis_tokens: [p_tokens, h_tokens]
        :return: token_ids, segments, valid_len
        """
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab["<pad>"]] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(tokens))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots (保留位置) for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, item):
        return (self.all_token_ids[item], self.all_segments[item],
                self.valid_lens[item], self.labels[item])

    def __len__(self):
        return len(self.all_token_ids)

class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder # 微调参数
        self.hidden = bert.hidden # bert 的 线性层，微调，只用到了这两个，mlm、nsp未用
        self.output = nn.Linear(256, 3) # 直接训练

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))

if __name__ == "__main__":
    devices = d2l.try_all_gpus()
    bert, vocab = load_pretrained_model('bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
                                        num_layers=2, dropout=0.1, max_len=512, devices=devices)

    batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
    data_dir = "../data/snli_1.0/"
    train_set = SNLIBERTDatset(read_snli(data_dir, True), max_len, vocab)
    test_set = SNLIBERTDatset(read_snli(data_dir, False), max_len, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, num_workers=num_workers)
    print(len(train_iter))

    net = BERTClassifier(bert)

    lr, num_epochs = 1e-4, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr) # 注意，这里的net.parameters()不包括bert.mlm, bert.nsp参数
    loss = nn.CrossEntropyLoss(reduction="none")
    d2l.train_epoch_ch3(net, train_iter, loss, trainer, num_epochs, devices)