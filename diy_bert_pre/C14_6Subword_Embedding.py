import collections



def get_max_freq_pair (token_freqs: dict)->tuple:
    """ 返回频次最高的连续符号对

    :param token_freqs:  the input dictionary,
        such as {'f a s t _': 4, 'f a s t e r _': 3, 't a l l _': 5, 't a l l e r _': 4}
    :return: the most frequent pair of consecutive symbols within a word
        assume that the Variable pairs = {('f', 'a'): 7, ('a', 's'): 7, ('s', 't'): 7, ('t', '_'): 4, ('t', 'e'): 3,
        ('e', 'r'): 7, ('r', '_'): 7, ('t', 'a'): 9, ('a', 'l'): 9, ('l', 'l'): 9, ('l', '_'): 5, ('l', 'e'): 4}),
        then the return is  ('t', 'a').
    """
    pairs = collections.defaultdict(int)            # 默认返回 0
    for token, freq in token_freqs.items():
        symbols = token.split()                        # ['f', 'a', 's', 't', '_']
        for i in range(len(symbols) - 1):           # 将元素两两配对，生成n-1对
            # Key of ' pairs ' is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i+1]] += freq # 两个key 默认转成 tuple
    return max(pairs, key=pairs.get) # dict().get() 函数返回指定键的值

def merge_symbols(max_freq_pair: tuple, token_freqs: dict, symbols: list)->dict:
    ''' merge the most frequent pair of consecutive symbols to produce new symbols

    :param max_freq_pair:  the most frequent pair of consecutive symbols within a word
        for example, ('t', 'a')
    :param token_freqs:  the input dictionary,
        such as: {'f a s t _': 4, 'f a s t e r _': 3, 't a l l _': 5, 't a l l e r _': 4}
    :param symbols: alphabet
        such as: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
    :return: the new dictionary, 新加入了最高频率的字母组合
    '''
    symbols.append(''.join(max_freq_pair)) # 字母表中添加新元素 ,例如”ab“
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs

def segment_BPE(tokens: list, symbols: list)->list:
    """

    :param tokens:  单词本
    :param symbols:  alphabet
    :return:   the longest possible subwords from the input argument symbols.
    """
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:  # 双指针找最大不重复匹配字符串
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
if __name__ == "__main__":
    symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
               '_', '[UNK]']

    raw_token_freqs = {"fast_": 4, "faster_": 3, "tall_": 5, "taller_": 4}
    token_freqs = {}
    for token, freq in raw_token_freqs.items():
        token_freqs[' '.join(list(token))] = freq


    num_merges = 10
    for i in range(num_merges):
        max_freq_pair = get_max_freq_pair(token_freqs)
        token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
        print(f'merge #{i + 1}:', max_freq_pair)


    tokens = ['talleset_', 'fatter_']
    print(segment_BPE(tokens, symbols))