# 9. The Dataset for Pretraining BERT

:label:`sec_bert-dataset`

总结

* 与PTB数据集相比，WikiText-2数据集保留了原来的**标点符号**、**大小写**和**数字**，并且比PTB数据集大了两倍多。
* 我们可以任意访问从WikiText-2语料库中的一对句子生成的预训练（遮蔽语言模型和下一句预测）样本。
* 训练bert时，两个预训练任务是同时进行的，所以我们需要把bert输入改成（NSP+MLM）输入
  * 先转NSP（50% 是下一句，50%概率是随机句子）
  * 再转MLM



To pretrain the BERT model as implemented in [8_bert.md](8_bert.md), we need to **generate the dataset** in the ideal format to facilitate the two pretraining tasks: **masked language modeling** and **next sentence prediction**. 为预训练任务做准备

- On one hand, the **original BERT model** is pretrained on the concatenation of two huge corpora BookCorpus and English Wikipedia (see [8_bert.md section 5](8_bert.md), making it hard to run for most readers of this book.
- On the other hand, the off-the-shelf （现成的） pretrained BERT model **may not fit** for applications from specific domains like medicine.
- Thus, it is getting popular to pretrain BERT on **a customized dataset**. 自己训练 预训练模型~

To facilitate the demonstration of BERT pretraining, we use a smaller corpus `WikiText-2` [[Merity et al., 2016]](https://d2l.ai/chapter_references/zreferences.html#merity-xiong-bradbury-ea-2016).

Comparing with the `PTB dataset` used for pretraining `word2vec` in [Section 14.3](https://d2l.ai/chapter_natural-language-processing-pretraining/word-embedding-dataset.html#sec-word2vec-data), `WikiText-2`

- (i) retains the original punctuation (保留了原始的标点符号), making it suitable for **next sentence prediction**;
- (ii) retains the original case （原始的大小写） and numbers （数字）;
- (iii) is over twice larger.（大了一倍以上）

```python
import os
import random
import torch
from d2l import torch as d2l
```

In the `WikiText-2` dataset, each line represents a paragraph where space is inserted between any punctuation and its preceding token. Paragraphs with at least two sentences are retained （保留至少有两句话的段落）. To split sentences, we only use the period (句号) as the delimiter （分隔符） for simplicity. We leave discussions of more complex sentence splitting techniques in the exercises （练习题） at the end of this section.

```python
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

```

## 9.1. Defining Helper Functions for Pretraining Tasks

In the following, we begin by implementing helper functions for the two BERT pretraining tasks: next sentence prediction and masked language modeling. These helper functions will be invoked later when **transforming** the raw text corpus **into** the dataset of the ideal format to pretrain BERT.

### 9.1.1 Generating the Next Sentence Prediction (NSP) Task

According to descriptions of [8_bert.md Section 8.5.2](8_bert.md), the `_get_next_sentence` function generates a training example for the binary classification task.

```python
#@save
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
```

The following function generates training examples for next sentence prediction from the input `paragraph` by invoking the `_get_next_sentence` function. Here `paragraph` is a list of sentences, where each sentence is a list of tokens. The argument `max_len` specifies the maximum length of a BERT input sequence during pretraining.

```python
#@save
def _get_nsp_data_from_paragraph(paragraph:list, paragraphs, vocab, max_len) -> list:
    nsp_data_from_paragraph = []
    for i in range(len(paragraph)):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i],  paragraph[i+1], paragraphs
        )
        #  考虑 1个 `<cls>` token 和 2个 `<sep> token
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### 9.1.2 Generating the Masked Language Modeling (MLM) Task

:label:`subsec_prepare_mlm_data`

In order to generate training examples for the MLM task from a BERT input sequence, we define the following `_replace_mlm_tokens` function.

In its inputs,

- `tokens` is a list of tokens representing a BERT input sequence,
- `candidate_pred_positions` is a list of token indices of the BERT input sequence **excluding those of special tokens** (special tokens are not predicted in the MLM task), and
- `num_mlm_preds` indicates the number of predictions (recall 15% random tokens to predict).

Following the definition of the MLM task in [8_bert Section 8.5.1](8_bert.md), at each prediction position, the input may be replaced by a special `“<mask>”` token or a random token, or remain unchanged. In the end, the function returns the input tokens after possible replacement, the token indices where predictions take place and labels for these predictions.

```python
#@save
def _replace_mlm_tokens(tokens:list, candidate_pred_positions:list, num_mlm_preds:int, vocab:type)->Tuple[list, List[tuple]]:
    """

    :param tokens: bert 句子
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

```

By invoking the aforementioned `_replace_mlm_tokens` function, the following function takes a BERT input sequence (`tokens`) as an input and returns indices of the input tokens (after possible token replacement as described in [8_bert Section 8.5.1](8_bert.md)) the token indices where predictions take place, and label indices for these predictions.

- `_replace_mlm_tokens` function：输入Bert input sequence，输出（mask后的句子，被mask单词的位置，mask位置的真实单词）

```python
#@save
def _get_mlm_data_from_tokens(tokens:list, vocab)->Tuple[list, list, list]:
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

```

## 9.2. Transforming Text into the Pretraining Dataset

Now we are almost ready to customize a `Dataset` class for pretraining BERT. 接下来就是要生成一个Dataset class了。

Before that, we still need to define a helper function `_pad_bert_inputs` **to append the special `“mask”` tokens to the inputs**.

Its argument `examples`contain the outputs **from** the helper functions `_get_nsp_data_from_paragraph` **and** `_get_mlm_data_from_tokens` for the two pretraining tasks.

```python
#@save
def _pad_bert_inputs(examples:List[tuple], max_len:int, vocab)->Tuple[list,list, list, list, list, list, list]:
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
            np.array(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'    )
        )
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens,
           all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)
```

Putting the helper functions for generating training examples of the two pretraining tasks, and the helper function for padding inputs together, we customize the following `_WikiTextDataset` class as the WikiText-2 dataset for pretraining BERT. By implementing the `__getitem__`function, we can arbitrarily access the pretraining (masked language modeling and next sentence prediction) examples generated from a pair of sentences from the WikiText-2 corpus. 用一个 class 合并两个任务数据集构建方法

The original BERT model uses WordPiece embeddings whose vocabulary size is 30000 [[Wu et al., 2016]](https://d2l.ai/chapter_references/zreferences.html#wu-schuster-chen-ea-2016). The tokenization method of WordPiece is a slight modification of the original byte pair encoding algorithm in [Section 14.6.2](https://d2l.ai/chapter_natural-language-processing-pretraining/subword-embedding.html#subsec-byte-pair-encoding). For simplicity, we use the `d2l.tokenize` function for tokenization. Infrequent tokens that appear less than five times are filtered out.

```python
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence **strings** representing a paragraph;
        # while output `paragraphs[i]` is a list of sentences representing a paragraph,
        # where each sentence is a list of **tokens**，
        # 即下面这个paragraphs：List[List[list]], 最外面的List是段落，里面的是句子，句子里面是单词
        paragraphs = [tokenize(paragraph, token='word') for paragraph in paragraphs]
        # 下面这个sentence：List[list],  他把句子与单词镶嵌的List合并为同一个了
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

By using the `_read_wiki` function and the `_WikiTextDataset` class, we define the following `load_data_wiki` to download and WikiText-2 dataset and generate pretraining examples from it.

```python
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

Setting the batch size to 512 and the maximum length of a BERT input sequence to be 64, we print out the shapes of a minibatch of BERT pretraining examples. Note that in each BERT input sequence, **10** (**64**×**0.15**) positions are predicted for the masked language modeling task.

```python
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

```
torch.Size([512, 64]) torch.Size([512, 64]) torch.Size([512]) torch.Size([512, 10]) torch.Size([512, 10]) torch.Size([512, 10]) torch.Size([512])
```

In the end, let us take a look at the vocabulary size. Even after filtering out infrequent tokens, it is still over twice larger than that of the PTB dataset.

```python
len(vocab)
```

20256

## Summary

* Comparing with the PTB dataset, the WikiText-2 dateset retains the original punctuation, case and numbers, and is over twice larger.
* We can arbitrarily access the pretraining (masked language modeling and next sentence prediction) examples generated from a pair of sentences from the WikiText-2 corpus.

## Exercises

1. For simplicity, the period is used as the only delimiter for splitting sentences. Try other sentence splitting techniques, such as the spaCy and NLTK. Take NLTK as an example. You need to install NLTK first: `pip install nltk`. In the code, first `import nltk`. Then, download the Punkt sentence tokenizer: `nltk.download('punkt')`. To split sentences such as `sentences = 'This is great ! Why not ?'`, invoking `nltk.tokenize.sent_tokenize(sentences)` will return a list of two sentence strings: `['This is great !', 'Why not ?']`.
2. What is the vocabulary size if we do not filter out any infrequent token?

[Discussions](https://discuss.d2l.ai/t/1496)
