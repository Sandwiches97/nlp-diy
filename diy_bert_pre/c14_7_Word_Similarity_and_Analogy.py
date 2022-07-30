import os
import torch
from torch import nn
from d2l_en.pytorch.d2l import torch as d2l

#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')

class TokenEmbedding:
    def __init__(self, embedding_name):
        self.idx2token, self.idx2vec = self._load_embedding(embedding_name)
        self.unknown_idx = 0
        self.token2idx = {token: idx for idx, token in
                          enumerate(self.idx2token)}

    def _load_embedding(self, embedding_name):
        idx2token, idx2vec = ["<unk>"], []
        dataDIr = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(dataDIr, "vec.txt"), 'r', encoding='utf-8') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(ele) for ele in elems[1:]]
                # skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx2token.append(token)
                    idx2vec.append(elems)
        idx2vec = [[0] * len(idx2vec[0])] + idx2vec
        return idx2token, torch.tensor(idx2vec)

    def __getitem__(self, tokens):
        indices = [self.token2idx.get(token, self.unknown_idx) for token in tokens]
        vecs = self.idx2vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx2token)

glove_6b50d = TokenEmbedding('glove.6b.50d')
print(f'the index of the word \'beautiful\' is :{glove_6b50d.token2idx["beautiful"]}, '
      f'and the word of index 123 is :{glove_6b50d.idx2token[123]}')

def knn(W, x, k):
    """ return topK b

    :param W: 词矩阵 （400000个单词，50维度）
    :param x: 查询的单词 （1个单词，50维度）
    :param k:  topK参数
    :return: 选出最接近的前k个单词
    """
    # k = torch.topk(cos, k=k)
    cos = torch.mv(W, x.reshape(-1, ))/(
        torch.sqrt(torch.sum(W*W, axis=1) + 1e-9) *
        torch.sqrt((x*x).sum())
    )
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]

def get_similar_tokens(query_token, k: int, embed: TokenEmbedding):
    topk, cos = knn(embed.idx2vec, embed[[query_token]], k+1)
    for i, c in zip(topk[1:], cos[1:]):
        print(f'cosine sim = {float(c): .3f}: {embed.idx2token[int(i)]}')

get_similar_tokens('chip', 3, glove_6b50d)


def get_analogy(token_a, token_b, token_c, embed:TokenEmbedding):
    """ a: b ~ c : ?(d), return d

    :param token_a:
    :param token_b:
    :param token_c:
    :param embed:
    :return:
    """
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs [0] + vecs[2]
    topk, cos = knn(embed.idx2vec, x, 1)
    return embed.idx2token[int(topk[0])]


