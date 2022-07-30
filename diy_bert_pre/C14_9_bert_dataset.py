
import numpy as np
import os
import random
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader
from d2l_en.pytorch.d2l import torch as d2l
from C14_8_BERT import get_tokens_and_segments
from diy_RNN.C8_2_Text_Preprocessing import tokenize, Vocab



#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir)->List[list]:
    """

    :param data_dir:
    :return:  (段落1， 段落2，...,)
    """
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones 大写字母转小写
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs

def _get_next_sentence(sentence:list, next_sentence:list, paragraphs:list):
    """
    the `_get_next_sentence` generates a training example for the binary classification task.
    :param sentence:
    :param next_sentence:
    :param paragraphs:
    :return:
    """
    if random.random() < 0.5:
        is_next = True
    else:
        # 'paragraphs' is a list of lists of lists (三重列表的嵌套)
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

def _get_nsp_data_from_paragraph(paragraph:list, paragraphs,  max_len) -> List[tuple]:
    nsp_data_from_paragraph = []
    for i in range(len(paragraph)-1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i],  paragraph[i+1], paragraphs
        )
        #  考虑 1个 `<cls>` token 和 2个 `<sep> token
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph

def _replace_mlm_tokens(tokens:list, candidate_pred_positions:list, num_mlm_preds:int, vocab:'Vocab')->Tuple[list, List[tuple]]:
    """

    :param tokens: BERT input sequence
    :param candidate_pred_positions: 预测的位置
    :param num_mlm_preds: 预测的数量
    :param vocab: 语料库 class 'C2_Text_Preprocessing.Vocab'
    :return:  a tuple: (mlm_input_tokens, pred_positions_and_labels)
    """
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the MLM task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None # 初始化变量
        # 80% of the time: replace the word with the "<mask>" token
        if random.random() < 0.8:
            masked_token = "<mask>"
        else:
            if random.random() < 0.5:
                masked_token = random.choice(vocab.idx_to_token)
            else:
                masked_token = tokens[mlm_pred_position]
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens:list, vocab:'Vocab')->Tuple[list, list, list]:
    """

    :param tokens:  Bert token
    :param vocab:
    :return:  a tuple (15% mask后的bert 输入，mask的位置，mask的真实token)
    """
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special token are not predicted in the MLM task
        if token in ["<CLS>", "<SEP>"]:
            continue
        else:
            candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the MLM task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [it[0] for it in pred_positions_and_labels]
    mlm_pred_labels = [it[0] for it in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

def _pad_bert_inputs(examples:List[tuple], max_len:int, vocab:'Vocab')->Tuple[list,list, list, list, list, list, list]:
    """  填 0 操作

    :param examples: MLM预训练的输入信息
    :param max_len: 最大长度
    :param vocab: class Vocab
    :return: 处理后的符合bert输入的  MLM预训练任务
    """
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (max_len - len(segments)), dtype='int32'))
        # `valid_len` excludes count of `<pad>` tokens
        valid_lens.append(np.array(len(token_ids), dtype='int32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        # Predictions of padded tokens will be filtered out in the loss via multiplication of 0 weights
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] *(max_num_mlm_preds - len(pred_positions)), dtype='float32')
        )
        all_mlm_labels.append(
            np.array(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int64')
        )
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens,
           all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)

class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs:List[list], max_len:int):
        # Input `paragraphs[i]` is a list of sentence **strings** representing a paragraph;
        # while output `paragraphs[i]` is a list of sentences representing a paragraph,
        # where each sentence is a list of **tokens**，
        # 即下面这个paragraphs：List[List[list]], 最外面的List是段落，里面的是句子，句子里面是单词
        paragraphs = [tokenize(paragraph, token='word') for paragraph in paragraphs]
        # 下面这个sentence：List[list],  他把句子与单词镶嵌的List合并为同一个了
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=[
            "<pad>", "<mask>", "<cls>", "<sep>"
        ])
        # get data for the NSP task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, max_len
            ))
        # get data for the MLM task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
                    for tokens, segments, is_next in examples]
        # pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels,
         self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, item):
        return (self.all_token_ids[item], self.all_segments[item], self.valid_lens[item],
                    self.all_pred_positions[item], self.all_mlm_weights[item], self.all_mlm_labels[item],
                    self.nsp_labels[item])

    def __len__(self):
        return len(self.all_token_ids)

def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab


if __name__ == "__main__":
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
         mlm_Y, nsp_y) in train_iter:
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
              pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
              nsp_y.shape)
        break

