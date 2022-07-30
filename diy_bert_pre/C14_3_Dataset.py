import math
import os
import random
import torch
from torch import utils
from d2l_en.pytorch.d2l import torch as d2l
from diy_RNN.C8_2_Text_Preprocessing import Vocab, count_corpus
from typing import Tuple
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')
#@save
def read_ptb():
    """
    将PTB数据集加载到文本行的列表中
    :return: a list

    just like : [['aer', 'banknote', 'berlitz', 'calloway', 'centrust', ...,], [..], [..]]
    """
    data_dir = d2l.download_extract('ptb')
    # Readthetrainingset.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

# sentences = read_ptb()
# f'# sentences数: {len(sentences)}'

# vocab = Vocab(sentences, min_freq=10) # 包含单词表的一些属性



def subsample(sentences:list, vocab:Vocab)->tuple:
    """
    Subsample high-frequency words

    :param sentences: a list including many sentences
    :param vocab: 单词表， Vocabulary for text
    :return:  a tuple: (sentences, counter:统计词频器)
        new sentences not included UNK and decreased the occur times of high-frequency words
    """
    # Exclude unknown tokens '<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    # return True if 'token' is kept during subsampling
    def keep(token):
        return (random.uniform(0, 1)<
                math.sqrt(1e-4 / counter[token] * num_tokens))
    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

# subsampled, counter = subsample(sentences, vocab)
# corpus = [vocab[line] for line in subsampled]
# print (' ')
# d2l.show_list_len_pair_hist(['origin', 'subsampled'], "# tockens per sentence",
#                             "count", sentences, subsampled)

def get_centers_and_contexts(corpus:list, max_window_size:int)->Tuple[list, list]:
    """
    Return center words and context words in skip-gram.

    :param corpus: 语料库 a List[list, list, ...]
    :param max_window_size: 最大窗口
    :return: a Tuple[list, List[list, list, ..., ]]
        centers = [2, 3, 12, ...],  shape = len(corpus)
        contexts = [[1,3,5], [...], ...], shape = (len(corpus), x), where x<=max window size
    """
    centers, contexts = [], []
    for line in corpus:
        # to form a "center words -- context word" pair, each sentence needs to
        # have at least 2 words
        if len(line)<2:
            continue
        centers += line
        for i in range(len(line)): # Context window centered at 'i'
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i-window_size), # max, min用于控制边界
                                 min(len(line), i + 1 + window_size)))
            indices.remove(i) # 从窗口里移除中心词
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

# tiny_dataset = [list(range(7)), list(range(7, 10))]
# print('datset', tiny_dataset)
# for center, context in zip(*get_centers_and_context(tiny_dataset, 2)):
#     print('center', center, "has contexts", context)

# all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
# f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'

class RandomGenerator:
    """
    Randomly draw among {1, ..., n} according to (n sampling weight, or 概率)
    an example like:
        generator = RandomGenerator([2, 3, 4]) # means 生成 1,2,3 的概率分别为：2/9, 3/9, 4/9
        [generator.draw() for _ in range(10)]
        ------------------------------------------------------------
        :return [3, 3, 2, 2, 2, 3, 2, 2, 3, 2]
    """
    def __init__(self, sampling_weights: list):
        # exclude
        self.population = list(range(1, len(sampling_weights)+1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0 # i 是输出索引

    def draw(self) -> int:
        if self.i == len(self.candidates): # 当输出了候选词长度的单词时，重新生成候选词
            # Cache 'k' random sampling results
            self.candidates = random.choices(
                self.population, weights=self.sampling_weights, k=10000
            ) # 将population里的元素按sampling_weighes的概率来随机生成，生成10000词
            self.i = 0
        self.i += 1
        return self.candidates[self.i-1]

# generator = RandomGenerator([2, 3, 4])
# print([generator.draw() for _ in range(10)])

def get_negatives(all_contexts, vocab, counter, K):
    """
    Return noise words in negative sampling

    :param all_contexts:
    :param vocab:
    :param counter:
    :param K:
    :return:
    """
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            #Noise words cannot to be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

# all_negatives = get_negatives(all_contexts, vocab, counter, 5)

def batchify(data):
    """
    Return a minibatch of examples for skip-gram with negative sampling.

    :param data: a list with length equal to the batch size, where each element is an example
        consisting of the center word `center`, its context words `context`, and its noise words `negative`
    :return: a minibatch that can be loaded for calculations during training, such as including the mask variable.
        (centers, contexts_negatives, mask, labels)

    an example like :
            x_1 = (1, [2, 2], [3, 3, 3, 3])
            x_2 = (1, [2, 2, 2], [3, 3])
            batch = batchify((x_1, x_2))

            names = ['centers', 'contexts_negatives', 'mask', 'labels']
            for name, data in zip(names, batch):
                print(name, '=', data)
            -------------------------------------------------------------
            centers = tensor([[1],
                    [1]])
            contexts_negatives = tensor([[2, 2, 3, 3, 3, 3],
                    [2, 2, 2, 3, 3, 0]])
            mask = tensor([[1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0]])
            labels = tensor([[1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0]])
    """
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [],[],[],[]
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0]*(max_len-cur_len)] # 不够长度的就补0
        masks += [[1]*cur_len + [0]*(max_len-cur_len)]
        labels += [[1]*len(context) + [0]*(max_len-len(context))]
    return (torch.tensor(centers).reshape((-1, 1)),
            torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))

# x_1 = (1, [2, 2], [3, 3, 3, 3])
# x_2 = (1, [2, 2, 2], [3, 3])
# batch = batchify((x_1, x_2))
#
# names = ['centers', 'contexts_negatives', 'mask', 'labels']
# for name, data in zip(names, batch):
#     print(name, '=', data)

def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """
    Download the PTB dataset and then load it into memory.

    :param batch_size:
    :param max_window_size: 用于get_centers_and_contexts()
    :param num_noise_words:
    :return:
    """
    num_workers = d2l.get_dataloader_workers() # return 4
    sentences = read_ptb()
    vocab = Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size
    )
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words
    )

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, item):
            return (self.centers[item], self.contexts[item],
                    self.negatives[item])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                            collate_fn=batchify)
    return data_iter, vocab

if __name__ == "__main__":
    data_iter, vocab = load_data_ptb(512, 5, 5)
    names = ['centers', 'contexts_negatives', 'mask', 'labels']
    for batch in data_iter:
        for name, data in zip(names, batch):
            print(name, "shape", data.shape)
        break