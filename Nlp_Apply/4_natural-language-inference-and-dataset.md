# 15.4 Natural Language Inference and the Dataset

:label:`sec_natural-language-inference-and-dataset`


- 自然语言推断研究“假设”是否可以从“前提”推断出来，其中两者都是文本序列。
- 在自然语言推断中，前提和假设之间的关系包括
  - 蕴涵关系
  - 矛盾关系
  - 中性关系
- 斯坦福自然语言推断（SNLI）语料库是一个比较流行的自然语言推断基准数据集。


In [Section 15.1](https://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.html#sec-sentiment), we discussed the problem of sentiment analysis. This task aims to **classify** a single text sequence **into** predefined categories, such as a set of sentiment polarities.

However, when there is a need to decide whether one sentence can be inferred form another (句子间的推理), or eliminate redundancy （消除冗余） by identifying sentences that are semantically equivalent （语义等价的句子）, knowing how to classify one text sequence is insufficient. Instead, we need to be able to reason over pairs of text sequences.

## 15.4.1 Natural Language Inference (NLI)

***Natural language inference (NLI)*** studies whether a *hypothesis （假设）* can be inferred from a  *premise （前提）* , where both are a text sequence. In other words, NLI determines the ==logical relationship== between a pair of text sequences. Such relationships usually fall into three types:

* *Entailment 蕴含* : the hypothesis can be inferred from the premise.
* *Contradiction 矛盾* : the negation of the hypothesis （假设的否定） can be inferred from the premise.
* *Neutral 中性* : all the other cases.

NLI is also known as the recognizing textual entailment task （识别文本蕴含任务）. For example, the following pair will be labeled as ***entailment*** because "showing affection 表达爱慕" in the hypothesis can be inferred from "hugging one another 拥抱" in the premise.

> Premise: Two women are hugging each other.

> Hypothesis: Two women are showing affection.

The following is an example of ***contradiction*** as "running the coding example" indicates "not sleeping" rather than "sleeping".

> Premise: A man is running the coding example from Dive into Deep Learning.

> Hypothesis: The man is sleeping.

The third example shows a ***neutrality*** relationship because neither "famous" nor "not famous" can be inferred from the fact that "are performing for us".

> Premise: The musicians are performing for us.

> Hypothesis: The musicians are famous.

NLI has been a central topic for understanding natural language. It enjoys wide applications ranging from information retrieval to open-domain question answering. To study this problem, we will begin by investigating a popular NLI benchmark dataset.

## 15.4.2 The Stanford Natural Language Inference (SNLI) Dataset

Stanford Natural Language Inference (SNLI) Corpus is a collection of over 500000 labeled English sentence pairs [[Bowman et al., 2015]](https://d2l.ai/chapter_references/zreferences.html#bowman-angeli-potts-ea-2015). We download and store the extracted SNLI dataset in the path `../data/snli_1.0`.

```python
import os
import re
import torch
from torch import nn
from d2l import torch as d2l

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### 15.4.2.1 Reading the Dataset

The original SNLI dataset contains much richer information than what we really need in our experiments. Thus, we define a function `read_snli` to only extract part of the dataset, then return lists of premises, hypotheses, and their labels.

```python
#@save
def read_snli(data_dir:str, is_train:bool)->Tuple[list, list, list]:
    """ Read the SNLI dataset into premises, hypotheses, and labels.

    :param data_dir:
    :param is_train:
    :return:
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
                             if is_train else 'snli_1.0_text.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

Now let us print the first 3 pairs of premise and hypothesis, as well as their labels ("0", "1", and "2" correspond to "entailment", "contradiction", and "neutral", respectively ).

```python
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

premise: A person on a horse jumps over a broken down airplane .
hypothesis: A person is training his horse for a competition .
label: 2
premise: A person on a horse jumps over a broken down airplane .
hypothesis: A person is at a diner , ordering an omelette .
label: 1
premise: A person on a horse jumps over a broken down airplane .
hypothesis: A person is outdoors , on a horse .
label: 0
The training set has about 550000 pairs, and the testing set has about 10000 pairs. The following shows that the three labels “entailment”, “contradiction”, and “neutral” are balanced in both the training set and the testing set.

```python
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```
[183416, 183187, 182764]
[3368, 3237, 3219]
### 15.4.2.2 Defining a Class for Loading the Dataset

Below we define a class for loading the SNLI dataset by inheriting from the `Dataset` class in Gluon.

- The argument `num_steps` in the class constructor specifies the length of a text sequence so that each minibatch of sequences will have the same shape. In other words, tokens after the first `num_steps` ones in longer sequence are trimmed, while special tokens `“<pad>”` will be appended to shorter sequences until their length becomes `num_steps`. (num_steps 指定了句子的长度，短的补零，长的截断)
- By implementing the `__getitem__`function, we can arbitrarily access the premise, hypothesis, and label with the index`idx`. （使用`[]`索引功能）

```python
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```
### 15.4.2.3 Putting All Things Together

Now we can invoke the `read_snli` function and the `SNLIDataset` class to download the SNLI dataset and return `DataLoader` instances for both training and testing sets, together with the vocabulary of the training set.

It is noteworthy that we must use the vocabulary constructed from the training set as that of the testing set. As a result, any new token from the testing set will be unknown to the model trained on the training set. 必须以训练集构造的vocabulary，作为测试集的vocabulary。

```python
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```
Here we set the batch size to 128 and sequence length to 50, and invoke the `load_data_snli` function to get the data iterators and vocabulary. Then we print the vocabulary size.

```python
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```
read 549367 examples
read 9824 examples





18678
Now we print the shape of the first minibatch. Contrary to sentiment analysis, we have two inputs `X[0]` and `X[1]` representing pairs of premises and hypotheses.

```python
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```
torch.Size([128, 50])
torch.Size([128, 50])
torch.Size([128])
## Summary

* Natural language inference studies whether a hypothesis can be inferred from a premise, where both are a text sequence.
* In natural language inference, relationships between premises and hypotheses include entailment, contradiction, and neutral.
* Stanford Natural Language Inference (SNLI) Corpus is a popular benchmark dataset of natural language inference.

## Exercises

1. Machine translation has long been evaluated based on superficial $n$-gram matching between an output translation and a ground-truth translation. Can you design a measure for evaluating machine translation results by using natural language inference?
2. How can we change hyperparameters to reduce the vocabulary size?

[Discussions](https://discuss.d2l.ai/t/1388)
