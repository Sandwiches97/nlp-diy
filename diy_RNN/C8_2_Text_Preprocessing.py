import collections
import re
from d2l_en.pytorch.d2l import torch as d2l

#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]



def tokenize(lines:list, token='word'):  #@save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines] # 空格分割
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


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

    # __getitem__  方法，使得Vocab实例 a, 能够通过 `a[idx]` 进行取值
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
        return self._token_freqs

def count_corpus(tokens):  #@save
    """Count token frequencies. 统计词频"""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char') ## 这里使用的是 字符 token
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab



def main():
    lines = read_time_machine()
    print(f'# text lines: {len(lines)}')
    print(lines[0])
    print(lines[10])


    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])

    ## 单词corpus
    vocab = Vocab(tokens)
    print(type(vocab), ' ', type(3))
    print(list(vocab.token_to_idx.items())[:10])

    ## 字符 corpus
    corpus, vocab = load_corpus_time_machine()
    print(len(corpus), len(vocab))

if __name__ == "__main__":
    main()