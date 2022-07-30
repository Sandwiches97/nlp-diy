# Language Models and the Dataset

:label:`sec_language_model`

- **语言模型**是自然语言处理的关键。
- **n-grams** 通过截断相关性，为处理长序列提供了一种实用的模型。
- **长序列**存在一个问题：它们很少出现或者从不出现。
- Zipf-law 支配着单词的分布，这个分布不仅适用于一元语法(unigrams)，还适用于其他元语法(n-grams)。
- 通过**拉普拉斯平滑法**可以有效地处理结构丰富而频率不足的低频词词组。
- 读取**长序列**的主要方式是**随机采样（Random Sampling）**和**顺序分区（Sequential Patitioning）**。在迭代过程中，后者可以保证来自两个相邻的小批量中的子序列在原始序列上也是相邻的。

In [section 3](3——), we see how to map text data into tokens, where these tokens can be viewed as a sequence of discrete observations, such as words or characters. Assume that the tokens in a text sequence of length $T$ are in turn $x_1, x_2, \ldots, x_T$. Then, in the text sequence, $x_t$($1 \leq t \leq T$) can be considered as the observation or label at time step $t$. Given such a text sequence, the goal of a **language model** is to estimate the joint probability of the sequence

$$
P(x_1, x_2, \ldots, x_T).

$$

$\color{red}\text{Language models}$ are incredibly useful. **For instance**, an ideal language model would be able to generate natural text just on its own, simply by drawing one token at a time $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$. **Quite unlike** the monkey using a typewriter, all text emerging from such a model would pass as natural language, e.g., English text. **Furthermore**, it would be sufficient for generating a meaningful dialog, simply by conditioning the text on previous dialog fragments. Clearly we are still very far from designing such a system, since it would need to *understand* the text rather than just generate grammatically sensible content.

尽管如此，语言模型依然是非常有用的。 例如，短语“to recognize speech”和“to wreck a nice beach”读音上听起来非常相似。 这种相似性会导致语音识别中的歧义，但是这很容易通过语言模型来解决， 因为第二句的语义很奇怪。 同样，在文档摘要生成算法中， “狗咬人”比“人咬狗”出现的频率要高得多， 或者“我想吃奶奶”是一个相当匪夷所思的语句， 而“我想吃，奶奶”则要正常得多。

## 3.1. Learning a Language Model

显而易见，我们面对的问题是如何对一个 Document， 甚至是一个词元序列 (sequence) 进行建模。 Suppose that we tokenize text data **at the word level**. We can take recourse to the analysis we applied to sequence models in [section 1](). Let us start by applying basic probability rules:

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).

$$

**For example**,the probability of **a text sequence** containing four words would be given as:

$$
P(\text{deep}, \text{learning}, \text{is}, \text{fun}) =  P(\text{deep}) P(\text{learning}  \mid  \text{deep}) P(\text{is}  \mid  \text{deep}, \text{learning}) P(\text{fun}  \mid  \text{deep}, \text{learning}, \text{is}).

$$

In order to compute the $\color{red}\text{Language model}$, we need to calculate the probability of words and the conditional probability of a word given the previous few words. **Such probabilities are essentially language model parameters.**  概率本质就是 $\color{red}\text{Language model}$ 的参数

Here, we assume that the training dataset is a large text corpus, such as all Wikipedia entries, [Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg), and all text posted on the Web. 训练数据集中词的概率可以根据 a given word 的 **relative word frequency** 来计算. **For example**, the estimate  $\hat{P}(deep)$ can be calculated as the probability of any sentence starting with the word "deep".

- 一种（稍稍不太精确的）方法是统计单词“deep”在数据集中的出现次数， 然后将其除以整个语料库中的单词总数。这种方法效果不错，特别是对于频繁出现的单词。
- 接下来，我们可以尝试估计

$$
\hat{P}(\text{learning} \mid \text{deep}) = \frac{n(\text{deep, learning})}{n(\text{deep})},

$$

where $n(x)$ and $n(x, x')$ are the number of occurrences of singletons and consecutive word pairs, respectively.

$\text{\colorbox{black}{\color{yellow}Unfortunately}}$, 由于连续单词对“deep learning”的出现频率要低得多， 所以估计这类单词正确的概率要困难得多。 **In particular**, for some unusual word combinations it may be tricky to find enough occurrences to get accurate estimates. Things take a turn for the worse for three-word combinations and beyond. There will be many plausible three-word combinations that we likely will not see in our dataset. Unless we provide some solution to assign such word combinations nonzero count, we will not be able to use them in a language model. If the dataset is small or if the words are very rare, we might not find even a single one of them.

A common strategy is to perform some form of  $\text{\colorbox{black}{\color{red}Laplace smoothing}}$ . The solution is to add a small constant to all counts. Denote by $𝑛$ the total number of words in the training set and **𝑚**m the number of unique words. This solution helps with singletons, e.g., via

$$
\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}

$$

Here $\epsilon_1,\epsilon_2$, and $\epsilon_3$ are hyperparameters. Take $\epsilon_1$ as an example: when $\epsilon_1 = 0$, no smoothing is applied; when $\epsilon_1$ approaches positive infinity, $\hat{P}(x)$ approaches the uniform probability $1/m$. The above is a rather primitive variant of what other techniques can accomplish [[Wood et al., 2011]](https://d2l.ai/chapter_references/zreferences.html#wood-gasthaus-archambeau-ea-2011).

**Unfortunately**, 这样的模型很容易变得无效:

- First, we need to store all counts.
- Second, this entirely $\text{\colorbox{black}{\color{magenta}ignores the meaning of the words}}$. **For instance**, "cat" and "feline" should occur in related contexts. It is quite difficult to adjust such models to additional contexts, whereas, deep learning based language models are well suited to take this into account.
- Last, long word sequences are almost certain to be novel, (长单词序列大部分都是未出现过的)

hence a model that simply counts the frequency of previously seen word sequences is bound to perform poorly there.

## 3.2. Markov Models and $n$-grams （n元语法）

Before we discuss solutions involving deep learning, we need some more **terminology and concepts**. Recall our discussion of $\text{\colorbox{black}{\color{red}Markov Models}}$ in [Section 8.1](1_Sequence_Models.md). Let us apply this to language modeling. A distribution over sequences satisfies the $\text{\colorbox{black}{\color{red}Markov property of first order}}$ if $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$. 阶数越高，对应的依赖关系就越长。 This leads to a number of approximations that we could apply to model a sequence:

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4), \text{一元语法，统计词频}\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}

$$

通常，涉及一个、两个和三个变量的概率公式分别被称为 “一元语法”（unigram）、“二元语法”（bigram）和“三元语法”（trigram）模型。 下面，我们将学习如何去设计更好的模型。

## 3.3. Natural Language Statistics

Let us see how this works on real data. We construct a vocabulary based on the time machine dataset as introduced in [Section 8.2](2_Text_Preprocessing.md) and print the top 10 most frequent words.

```python
import random
import torch
from d2l import torch as d2l
```

```python
tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
```

```
[('the', 2261),
('i', 1267),
('and', 1245),
('of', 1155),
('a', 816),
('to', 695),
('was', 552),
('in', 541),
('that', 443),
('my', 440)]
```

As we can see, (**the most popular words are** 没什么用) actually quite boring to look at. They are often referred to as (***stop words***) and thus filtered out （一般会被过滤掉）.  **Nonetheless**, 它们本身仍然是有意义的，我们仍然会在模型中使用它们. **Besides**, it is quite clear that the word frequency decays rather rapidly （衰减很快）. The $10^{\mathrm{th}}$ most frequent word is less than $1/5$ as common as the most popular one. To get a better idea, we [**plot the figure of the word frequency**].

```python
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

![../_images/output_language-models-and-dataset_789d14_18_0.svg](https://zh.d2l.ai/_images/output_language-models-and-dataset_789d14_18_0.svg)

通过此图我们可以发现：词频以一种明确的方式迅速衰减。 将前几个单词作为例外消除后，剩余的所有单词大致遵循双对数坐标图上的一条直线。 This means that words satisfy *Zipf's law*, which states that the frequency $n_i$ of the $i^\mathrm{th}$ most frequent word is:

$$
n_i \propto \frac{1}{i^\alpha},\tag{8.3.7}

$$

which is equivalent to

$$
\log n_i = -\alpha \log i + c,\tag{8.3.8}

$$

where $\alpha$ is the exponent that characterizes the distribution and $c$ is a constant. 这告诉我们想要通过计数统计和平滑来建模单词是不可行的， 因为这样建模的结果会大大高估尾部单词的频率，也就是所谓的不常用单词。

But [**what about the other word combinations, such as bigrams, trigrams**], and beyond?

Let us see whether the **bigram** frequency behaves in the same manner as the unigram frequency.

```python
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

```
[(('of', 'the'), 309),
(('in', 'the'), 169),
(('i', 'had'), 130),
(('i', 'was'), 112),
(('and', 'the'), 109),
(('the', 'time'), 102),
(('it', 'was'), 99),
(('to', 'the'), 85),
(('as', 'i'), 78),
(('of', 'a'), 73)]
```

这里值得注意：在十个最频繁的词对中，有九个是由两个停用词组成的， 只有一个与“the time”有关。 我们再进一步看看 trigrams 的频率是否表现出相同的行为方式。

```python
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

```
[(('the', 'time', 'traveller'), 59),
(('the', 'time', 'machine'), 30),
(('the', 'medical', 'man'), 24),
(('it', 'seemed', 'to'), 16),
(('it', 'was', 'a'), 15),
(('here', 'and', 'there'), 15),
(('seemed', 'to', 'me'), 14),
(('i', 'did', 'not'), 14),
(('i', 'saw', 'the'), 13),
(('i', 'began', 'to'), 13)]
```

最后，我们直观地对比三种模型中的词元频率：一元语法、二元语法和三元语法。

```python
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

![../_images/output_language-models-and-dataset_789d14_54_0.svg](https://zh.d2l.ai/_images/output_language-models-and-dataset_789d14_54_0.svg)

This figure is quite exciting for a number of reasons.

- First, beyond unigram words, **sequences of words**(单词序列也遵循 Zipf 定律) also appear to be following Zipf's law, albeit with a smaller exponent $\alpha$ in [(8.3.7)](), depending on the sequence length.
- Second, 词表中 n 元组的数量并没有那么大，这说明语言中存在相当多的结构
- Third, 很多 n 元组很少出现，这使得拉普拉斯平滑非常不适合语言建模

Instead, we will use **deep learning based models**.

## 3.4. Reading Long Sequence Data 读取长序列

Since sequence data are by their very nature sequential, we need to address the issue of processing it. We did so in a rather ad-hoc (很特别的方式) manner [Section 8.1](1_Sequences_Models.md). When sequences get too long to be processed by models all at once, we may wish to split such sequences for reading. 模型太长的时候，我们希望能把这样的序列拆分了。

在介绍该模型之前，我们看一下总体策略。 假设我们将使用神经网络来训练语言模型， 模型中的网络一次处理具有预定义长度 （例如 $n$ 个时间步）的一个小批量序列。 现在的问题是如何随机生成一个小批量数据的特征和标签以供读取。

首先，由于文本序列可以是任意长的， 例如整本《时光机器》（ *The Time Machine* ）, we can **partition** such a long sequence **into** `subsequences` with the same number of time steps. When training our neural network, a minibatch of such subsequences will be fed into the model.

Suppose that the network processes a subsequence of $n$ time steps at a time. [Fig. 8.1.3]() shows all the different ways to obtain subsequences from an original text sequence, where $n=5$ and a token at each time step corresponds to a character. 请注意，因为我们可以选择任意偏移量 (offset) 来指示初始位置，所以我们有相当大的自由度。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/timemachine-5gram.svg" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 8.1.3 Different offsets lead to different subsequences when splitting up text.
  	</div>
</center>

Hence, which one should we pick from [Fig. 8.3.1](https://d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html#fig-timemachine-5gram)?

事实上，他们都一样的好。 然而，如果我们只选择一个偏移量， 那么用于训练网络的、所有可能的子序列的覆盖范围将是有限的。 因此，我们可以从随机偏移量 (a random offset) 开始划分序列， 以同时获得 *覆盖性* （coverage）和 *随机性* （randomness）。 下面，我们将描述如何实现

- *随机采样* （random sampling）和
- *顺序分区* （sequential partitioning）策略。

### 3.4.1Random Sampling 随机采样

( **In $\text{\colorbox{white}{\color{red}random sampling}}$, each example is a subsequence arbitrarily captured on the original long sequence.** ) The subsequences from two adjacent random minibatches during iteration are not necessarily adjacent on the original sequence. (就是将batch size**内部**打包的时候，不要按顺序打包，而是随机打包)

For **$\text{\colorbox{black}{\color{red}language modeling}}$**, the target is to $\text{\colorbox{black}{\color{yellow}predict the next token}}$ based on what tokens we have seen so far, hence the `labels` are the original sequence, **$\text{\colorbox{black}{\color{yellow}shifted by one token}}$**.

下面的代码每次可以从数据中随机生成一个小批量。 Here, the argument

- `batch_size` specifies the number of subsequence examples in each minibatch and
- `num_steps` is the predefined number of time steps in each subsequence.

```python
def seq_data_iter_random(corpus, batch_size, num_steps)->Tuple:  #@save
    """ 下面的代码每次可以从数据中随机生成一个小批量。

    :param corpus: a list
    :param batch_size: 每个小批量中子序列样本的数目
    :param num_steps: 每个子序列中预定义的时间步数
    :return:
    """
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签, idx_Y = (idx_X + 1)
    num_subseqs = (len(corpus) - 1) // num_steps        # 子序列的个数
    # The starting indices 起始索引 for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices 包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
```

Let us [**manually generate a sequence from 0 to 34.**] We assume that the `batch size` and `numbers of time steps` are 2 and 5, respectively. This means that we can generate $\lfloor (35 - 1) / 5 \rfloor= 6$ `feature-label subsequence pairs`. With a minibatch size of 2, we only get 3 minibatches.

```python
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

```
X:  tensor([[24, 25, 26, 27, 28],
[ 9, 10, 11, 12, 13]])
Y: tensor([[25, 26, 27, 28, 29],
[10, 11, 12, 13, 14]])
X:  tensor([[29, 30, 31, 32, 33],
[ 4,  5,  6,  7,  8]])
Y: tensor([[30, 31, 32, 33, 34],
[ 5,  6,  7,  8,  9]])
X:  tensor([[14, 15, 16, 17, 18],
[19, 20, 21, 22, 23]])
Y: tensor([[15, 16, 17, 18, 19],
[20, 21, 22, 23, 24]])
```

### Sequential Partitioning 顺序分区

In addition to random sampling of the original sequence, [ 我们还可以保证两个相邻的小批量中的子序列在原始序列上也是相邻的 ] 。 这种策略在基于小批量的迭代过程中保留了拆分的子序列的顺序, hence is called $\text{\colorbox{white}{\color{red}sequential partitioning}}$.

```python
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

Using the same settings, let us [ **print features `X` and labels `Y` for each minibatch** ] of subsequences read by sequential partitioning. 通过将它们打印出来可以发现： 迭代期间来自两个相邻的小批量中的子序列在原始序列中确实是相邻的。

```python
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

```
X:  tensor([[ 0,  1,  2,  3,  4],
[17, 18, 19, 20, 21]])
Y: tensor([[ 1,  2,  3,  4,  5],
[18, 19, 20, 21, 22]])
X:  tensor([[ 5,  6,  7,  8,  9],
[22, 23, 24, 25, 26]])
Y: tensor([[ 6,  7,  8,  9, 10],
[23, 24, 25, 26, 27]])
X:  tensor([[10, 11, 12, 13, 14],
[27, 28, 29, 30, 31]])
Y: tensor([[11, 12, 13, 14, 15],
[28, 29, 30, 31, 32]])
```

Now we wrap the above two sampling functions to a `class`

```python
class SeqDataLoader:  #@save
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

[**Last, we define a function `load_data_time_machine` that returns both the data iterator and the vocabulary**], so we can use it similarly as other other functions with the `load_data` prefix, such as `d2l.load_data_fashion_mnist` defined [Section 3.5](https://d2l.ai/chapter_linear-networks/image-classification-dataset.html#sec-fashion-mnist).

```python
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

## Summary

* Language models are key to natural language processing.
* $n$-grams provide a convenient model for dealing with long sequences by truncating the dependence.
* Long sequences suffer from the problem that they occur very rarely or never.
* Zipf's law governs the word distribution for not only unigrams but also the other $n$-grams.
* There is a lot of structure but not enough frequency to deal with infrequent word combinations efficiently via Laplace smoothing.
* The main choices for reading long sequences are random sampling and sequential partitioning. The latter can ensure that the subsequences from two adjacent minibatches during iteration are adjacent on the original sequence.

## Exercises

1. Suppose there are $100,000$ words in the training dataset. How much word frequency and multi-word adjacent frequency does a four-gram need to store?
2. How would you model a dialogue?
3. Estimate the exponent of Zipf's law for unigrams, bigrams, and trigrams.
4. What other methods can you think of for reading long sequence data?
5. Consider the random offset that we use for reading long sequences.
   1. Why is it a good idea to have a random offset?
   2. Does it really lead to a perfectly uniform distribution over the sequences on the document?
   3. What would you have to do to make things even more uniform?
6. If we want a sequence example to be a complete sentence, what kind of problem does this introduce in minibatch sampling? How can we fix the problem?

[Discussions](https://discuss.d2l.ai/t/118)
