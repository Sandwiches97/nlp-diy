# Text Preprocessing

:label:`sec_text_preprocessing`

- 文本是序列数据的一种最常见的形式之一。
- 为了对文本进行预处理，我们通常
  - step1 Token: 将文本拆分为词元，（有两种：1）words；2）char）
  - step2 Corpus: 构建词表将词元字符串映射为数字索引 （根据词频），
  - step3: 并将文本数据转换为词元索引以供模型操作。

We have reviewed and evaluated statistical tools and prediction challenges for sequence data. Such data can take many forms. Specifically, as we will focus on in many chapters of the book, text is one of the most popular examples of sequence data. For example, an article can be simply viewed as a sequence of words, or even a sequence of characters. To facilitate our future experiments with sequence data, we will dedicate this section to explain common **preprocessing steps for text**. Usually, these steps are:

1. Load text as `strings` into memory.
2. **Split** strings **into** `tokens` (e.g., words and characters).
3. Build `a table of vocabulary` to **map** the split tokens **to** numerical indices.
4. **Convert** text **into** `sequences of numerical indices` so they can be manipulated by models easily.

```python
import collections
import re
from d2l_en.pytorch.d2l import torch as d2l
```

## Reading the Dataset

To get started we load text from H. G. Wells' [*The Time Machine*](http://www.gutenberg.org/ebooks/35). This is a fairly small corpus of just over 30000 words, but for the purpose of what we want to illustrate this is just fine. More realistic document collections contain many billions of words. The following function ( **reads the dataset into a list of text lines** ), where each line is a string. For simplicity, here we ignore punctuation and capitalization.

```python
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

```

## Tokenization （字符或者单词）

The following `tokenize` function takes a list (`lines`) as the input, where each element is a text sequence (e.g., a text line). [ **Each text sequence is split into a list of tokens** ]. A *token* is the basic unit in text. In the end, a list of token lists are returned, where each token is a string.

```python
def tokenize(lines:list, token='word'):  #@save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines] # 空格分割
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)
```

## Vocabulary

The string type of the token is inconvenient to be used by models, which take numerical inputs. Now let us [ **build a dictionary, often called *vocabulary* as well, to map string tokens into numerical indices starting from 0 （用于将token:str -> index:int）** ].

- To do so, we first count the unique tokens in all the documents from the training set, namely a  `corpus`(语料) , and then assign a numerical index to each unique token according to its **frequency**. （corpus 用于记录词频）
- **Rarely appeared tokens are often removed** to reduce the complexity. （corpus 删除了 频率低于 `min_freq默认为0` 的token）
- Any token that does **not exist** in the corpus or **has been removed** is mapped into a special unknown token `<unk>`. （语料库中**不存在的** 和 **已经被删除**的 token，用`<unk>`代替）
- We optionally add a list of `reserved tokens`, such as
  - `<pad>`for padding,
  - `<bos>`to present the beginning for a sequence, and
  - `<eos>` for the end of a sequence.

```python
class Vocab:  #@save
    """Vocabulary for text.
    维护了1个 list, 1个字典, 用来索引
        - idx_to_token: list
            使用方法，例如，取第5个单词，Vocab().idx_to_token[5]
        - token_to_idx: 字典{}, 
            使用方法，例如，将一个词或者一个list的词 tokens 转换为索引， Vocab()[tokens]
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    # __getitem__  方法，使得Vocab实例能够通过 `[idx]`进行取值
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)): # isinstance(a, `type b`), return type(a)==b
            return self.token_to_idx.get(tokens, self.unk) # {}.get(key, default value) 方法，查询key, 如没有则返回 default value
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freq
```

## Putting All Things Together

Using the above functions, we [ **package everything into the `load_corpus_time_machine` function** ], which returns (`corpus`: a list of token indices, and `vocab`: the vocabulary of the time machine corpus). The modifications we did here are:

- (i) we tokenize text into characters, not words, to simplify the training in later sections;
- (ii) `corpus` is a single list, not a list of token lists, since each text line in the time machine dataset is not necessarily a sentence or a paragraph. 注意，这里用的是**字符corpus**，而不是词元corpus

```py
def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```
