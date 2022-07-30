# 8.Bidirectional Encoder Representations from Transform (BERT)

- word2vec和GloVe等词嵌入模型与**上下文无关**。它们将相同的预训练向量赋给同一个词，而不考虑词的上下文（如果有的话）。它们很难处理好自然语言中的一词多义或复杂语义。
- 对于**上下文敏感**的词表示，如ELMo和GPT，词的表示依赖于它们的上下文。但是二者都有缺点：
  - ELMo对上下文进行双向编码，但使用特定于任务的架构（然而，为每个自然语言处理任务设计一个特定的体系架构实际上并不容易）；
  - 而GPT是任务无关的，但是从左到右编码上下文。
- BERT结合了这两个方面的优点：它对上下文进行双向编码，并且需要对大量自然语言处理任务进行最小的架构更改。
- BERT输入序列的嵌入是**token embedding**、**sentence embedding**和 **learnable position embedding** 的和。
- 预训练包括两个任务：
  - **masked language modeling (MLM)**，编码双向上下文来表示单词（15%（80：10：10））
  - **next sentence prediction (NSP)**，显式地建模文本对之间的逻辑关系。

We have introduced several word embedding models for natural language understanding. **After pretraining**, the output can be thought of as **a matrix** where each row is a vector that represents a word of a predefined vocabulary. **In fact**, these word embedding models are all **context-independent** . Let us begin by illustrating this property.

## 8.1. From Context-Independent to Context-Sensitive

we know the `word2vec` and `GloVe` method both assign the same pretrained vector to the same word **regardless of the context of the word.** （二者都没有考虑上下文）

Formally, a context-independent representation of any token $x$ is a function $f(x)$ that **only takes $x$ as its input**. Given the abundance of polysemy and complex semantics in natural languages, context-independent representations have **obvious limitations**. **For instance**, the word “crane” in contexts “a crane is flying” and “a crane driver came” has completely different meanings; thus, **the same word may be assigned different representations depending on contexts.**（ 一个词可以根据上下文被赋予不同的含义）

This motivates the development of **context-sensitive** word representations, where representations of words depend on their contexts. Hence, a context-sensitive representation of token x is **a function $f(x,c(x))$ depending on both $x$ and its context $c(x)$**. Popular context-sensitive representations include **TagLM** (language-model-augmented sequence tagger) [[Peters et al., 2017b]](https://d2l.ai/chapter_references/zreferences.html#peters-ammar-bhagavatula-ea-2017), **CoVe** (Context Vectors) [[McCann et al., 2017]](https://d2l.ai/chapter_references/zreferences.html#mccann-bradbury-xiong-ea-2017), and **ELMo** (Embeddings from Language Models) [[Peters et al., 2018]](https://d2l.ai/chapter_references/zreferences.html#peters-neumann-iyyer-ea-2018).

**For example**, by taking the entire sequence as the input,

- **ELMo** is a function that assigns a representation to each word from the input sequence.
  - **Specifically**, ELMo **combines all** the intermediate layer representations from `pretrained bidirectional LSTM` **as** the output representation.
  - **Then** the ELMo representation will be added to a downstream task’s (下游任务是CV、NLP等，用于评估通过自监督学习学习到的特征的质量，例如CV中的目标检测、语义分割等等) existing supervised model as additional features, such as by concatenating ELMo representation and the original representation (e.g., GloVe) of tokens in the existing model.
  - **On one hand**, all the weights in the pretrained bidirectional LSTM model **are frozen after** ELMo representations are added.
  - **On the other hand**, the existing supervised model is specifically customized for a given task.
  - Leveraging different best models for different tasks at that time, adding ELMo improved the state of the art across six natural language processing tasks: sentiment analysis, natural language inference, semantic role labeling, coreference resolution, named entity recognition, and question answering.

## 8.2. From Task-Specific (特定任务) to Task-Agnostic (任务不可知)

Although **ELMo** has significantly improved solutions to a diverse set of natural language processing tasks, each solution still hinges on a **task-specific** architecture.  **However** , it is practically non-trivial to craft a specific architecture for every natural language processing task.

The **GPT (Generative Pre-Training) model** represents an effort in designing a general **task-agnostic** model for context-sensitive representations [[Radford et al., 2018]](https://d2l.ai/chapter_references/zreferences.html#radford-narasimhan-salimans-ea-2018). Built on a transformer decoder, GPT pretrains a language model that will be used to represent text sequences. When applying GPT to a downstream task, the output of the language model **will be fed into an added linear output layer** to predict the label of the task.

In sharp contrast to **ELMo** that freezes parameters of the pretrained model (与ELMo 冻结参数相比), **GPT fine-tunes all the parameters** in the pretrained transformer decoder during supervised learning of the downstream task. **GPT** was evaluated on twelve tasks of natural language inference, question answering, sentence similarity, and classification, and improved the state of the art in nine of them with minimal changes to the model architecture.

**However** , due to the **autoregressive nature** of language models,  **GPT only looks forward (left-to-right)** .

- GPT虽然做到了Task-Agnostic，但是他是Context-Independent，即面对下面情况时：
  - “i went to the **bank** to deposit cash”，这个bank：**银行**
  - “i went to the **bank** to sit down”，这个bank：**河岸**
  - GPT will return the **same representation** for “bank”, though it has different meanings.

## 8.3. BERT: Combining the Best of Both Worlds (两全其美的方案)

As we have seen,

- **ELMo** encodes context bidirectionally **but** uses task-specific architectures;
- while **GPT** is task-agnostic **but** encodes context left-to-right.
- Combining the best of both worlds, **BERT (Bidirectional Encoder Representations from Transformers)** encodes context bidirectionally and requires minimal architecture changes for a wide range of NLP tasks [[Devlin et al., 2018]](https://d2l.ai/chapter_references/zreferences.html#devlin-chang-lee-ea-2018).

Using `a pretrained transformer encoder`, **BERT** is able to represent any token based on its bidirectional context. During supervised learning of downstream tasks, BERT **is similar to** GPT in two aspects.

- **First**, BERT representations will be fed into an added output layer, with **minimal changes to the model architecture** depending on nature of tasks, such as **predicting for every token** $\textbf{vs.}$ **predicting for the entire sequence.**
- **Second**, all the parameters of the pretrained transformer encoder are **fine-tuned**, while the additional output layer will be trained from scratch.  (从头开始训练)

[Fig. 8.1]() depicts the differences among ELMo, GPT, and BERT.

![]()

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/elmo-gpt-bert.svg"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig 8.1 A comparison of ELMo, GPT, and BERT.
  	</div>
</center>

**BERT** further improved the state of the art on eleven NLP tasks under broad categories of ：

- (i) single text classification (单一文本分类，e.g., sentiment analysis),
- (ii) text pair classification (文本对分类，e.g., natural language inference),
- (iii) question answering, 问答系统
- (iv) text tagging (文本标记，e.g., named entity recognition).

All proposed in 2018, from context-sensitive ELMo to task-agnostic GPT and BERT, conceptually simple yet empirically powerful pretraining of deep representations for natural languages have revolutionized solutions to various NLP tasks.

In the rest of this chapter, we will dive into the **pretraining of BERT**. When natural language processing applications are explained in [Section 15](https://d2l.ai/chapter_natural-language-processing-applications/index.html#chap-nlp-app), we will illustrate fine-tuning of BERT for downstream applications.

```python
import torch
from torch import nn
from d2l import torch as d2l
```

## 8.4. Input Representation

In NLP, some tasks (e.g., sentiment analysis) take **single text** as the input, while in some other tasks (e.g., natural language inference), the input is a **pair of text sequences**. (输入的形式多样)

The **BERT** input sequence unambiguously (无歧义地) represents **both** `single text` **and** `text pairs`.

- **In the single text** ： the BERT input sequence is the concatenation of the special classification token ``<cls> ``, tokens of a text sequence, and the special separation token ``<sep>``. （`["<cls>", "i", "love", "u", "<sep>"]`）
- **In the text pairs** ： the BERT input sequence is the **concatenation** of ``<cls>``, tokens of the first text sequence, ``<sep>``, tokens of the second text sequence, and ``<sep>`` . （`["<cls>", "i", "love", "u", "<sep>", "but", "you", "hate", "me", "<sep>"]`）

We will consistently **distinguish** the terminology “BERT input sequence” **from** other types of “sequences”. For instance, one **BERT input sequence** may include **either** one *text sequence ***or** two  *text sequences* .

To distinguish **text pairs**, the learned segment embeddings $e_A$ and $e_B$ are added to the token embeddings of the first sequence and the second sequence, respectively. For **single text** inputs, only $e_A$ is used.

The following `get_tokens_and_segments`

- **input**: **either** one sentence **or** two sentences,
- **returns**:
  - tokens of the BERT input sequence
  - their corresponding segment IDs.

```python
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT chooses the `transformer encoder` as its bidirectional architecture.

- Common in the transformer encoder, **positional embeddings** are added at every position of the BERT input sequence.
- **However**, different from the original transformer encoder, BERT uses **learnable positional embeddings**.

To sum up, [Fig. 8.2]() shows that the embeddings of the `BERT input sequence` are the sum of the `token embeddings`, `segment embeddings`, and `positional embeddings`.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/bert-input.svg"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig 8.2 The embeddings of the BERT input sequence are the sum of the token embeddings, segment embeddings, and positional embeddings.
  	</div>
</center>

The following `BERTEncoder` class **is similar to** the `TransformerEncoder` class as implemented in [Section 10.7](../diy_Attention/7_Transformer.md). （输出shape跟transformer一致）

**Different from** `TransformerEncoder `(与 Transformer encoder 输入的不同之处有), `BERTEncoder` uses

- **segment embeddings** （句子编码，nn.Embedding：让Bert的`<CLS>`知道它学到的东西是来自两个句子的）
- **learnable positional embeddings**. （可学习 位置编码，就是一个torch.Parameter）

```python
#@save
class BERTEncoder(nn.Module):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

举例子：

Suppose that the vocabulary size is 10000. To demonstrate forward inference of `BERTEncoder`, let us create an instance of it and initialize its parameters.

```python
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
```

We define `tokens` to be 2 BERT input sequences of length 8, where each token is an index of the vocabulary. The forward inference of `BERTEncoder` with the input `tokens` returns the encoded result where each token is represented by a vector whose length is predefined by the hyperparameter `num_hiddens`. This hyperparameter is usually referred to as the ***hidden size*** (number of hidden units) of the **transformer encoder**.

```python
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

torch.Size([2, 8, 768])

## 8.5. Pretraining Tasks

🏷`subsec_bert_pretraining_tasks`

The **forward inference** of `BERTEncoder` **gives** the BERT representation of **each token of the input text** and the **inserted special tokens** `“<cls>”` and `“<seq>”`. Next, we will use these representations to compute the loss function for pretraining BERT.

The pretraining is composed of the following two tasks:

- masked language modeling
- next sentence prediction.

### 8.5.1. Masked Language Modeling (解决双向context的问题)[¶](https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html#masked-language-modeling "Permalink to this headline")

As illustrated in [Section 8.3](../diy_RNN/3_language_models_and_dataset.md), a **language model** predicts a token using the context on its **left**.

**To encode context bidirectionally** for representing each token, **BERT** randomly masks tokens and uses tokens from the bidirectional context to predict the masked tokens in a self-supervised fashion.(随机遮掩 tokens, 并使用来自双向上下文的词元以自监督的方式**预测** masked tokens) This task is referred to as a  **masked language model** .

具体而言：（主要是针对输入进行操作，选择15%的token，将其替换，其中它们80%的时间(概率)是mask、10%的时间是正确单词、10%时间是随机单词）

In this pretraining task, 15% of tokens will be selected at random as the masked tokens for prediction. To predict a masked token without cheating by using the label,

- one straightforward approach is to always replace it with a **special `“mask”` token** in the BERT input sequence. **However**, the artificial special token `“<mask>”` will **never appear in fine-tuning**.
- To avoid such a mismatch between pretraining and fine-tuning, if a token is masked for prediction (e.g., `"great"` is selected to be masked and predicted in `"this movie is great"`), in the input it will be replaced with:
  - a special `“<mask>”` token for 80% of the time (e.g., `"this movie is great"` becomes `"this movie is <mask>"`);
  - a random token for 10% of the time (e.g., `"this movie is great"` becomes `"this movie is drink"`);
  - the unchanged label token for 10% of the time (e.g., `"this movie is great"` becomes `"this movie is great"`).

Note that for 10% of 15% time a random token is inserted. This **occasional noise** encourages BERT to be less biased (不那么偏向) towards the **masked token** (especially when the label token remains unchanged) in its bidirectional context encoding.

下游任务的代码实现：（不包含mask的实现，mask的实现见[9_bert数据集构建](./9_bert_dataset.md)）

We implement the following `MaskLM` class to predict masked tokens in the masked language model task of BERT pretraining. The prediction uses a one-hidden-layer MLP (`self.mlp`).

In forward inference, it takes **two inputs**:

- the encoded result of `BERTEncoder`
- the token positions for prediction.

The **output** is the prediction results at these positions. shape = `(batch size, 句子长度, vocab 长度)`

```python
#@save
class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        """

        :param X:
            the encoded result of BERTEncoder
        :param pred_positions:
            the token positions for prediction.
        :return:
            the prediction results at these positions. (batch size, 句子长度, vocab 长度)
        """
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions] # 被masked掉的 (batchsize * num_pred_positions) 个单词
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

例子：本例子是下游任务，并没有给出mask的具体生成方式，具体见[9_bert数据集构建](./9_bert_dataset.md)

To demonstrate the forward inference of `MaskLM`, we create its instance `mlm` and initialize it. Recall that `encoded_X` from the forward inference of `BERTEncoder` represents 2 BERT input sequences.

We define `mlm_positions` as the 3 indices to predict in either BERT input sequence of `encoded_X`(定义为在TransformerEncoder输出的任一输入序列中预测的3个指示). The forward inference of `mlm`

- returns prediction results `mlm_Y_hat` at all the masked positions `mlm_positions` of `encoded_X`.

For each prediction, the size of the result is equal to the vocabulary size.

```python
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

torch.Size([2, 3, 10000])
With the ground truth labels `mlm_Y` of the predicted tokens `mlm_Y_hat` under masks, we can calculate the **cross-entropy loss** of the masked language model task in BERT pretraining.

```python
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

torch.Size([6])

### 8.5.2 Next Sentence Prediction (解决text pairs对之间的逻辑问题)

🏷`subsec_nsp`

**Although** masked language modeling is able to encode bidirectional context for representing words, it does not explicitly model the logical relationship between text pairs.

To help understand the relationship between two text sequences, BERT considers **a binary classification task**,  **next sentence prediction** , in its pretraining.

- When generating sentence pairs for pretraining, for half of the time they are indeed consecutive sentences with the label "True"; 一半的真
- while for the other half of the time the second sentence is randomly sampled from the corpus with the label "False". 一半的假

代码实现：

The following `NextSentencePred` class uses a one-hidden-layer MLP to predict whether the second sentence is the next sentence of the first in the BERT input sequence. Due to self-attention in the transformer encoder, the BERT representation of the special token `“<cls>”` encodes both the two sentences from the input. Hence, the output layer (`self.output`) of the MLP classifier takes `X`as the input, where `X`is the output of the MLP hidden layer whose input is the encoded `“<cls>”` token.

- input：`TransformerEncoder的输出X.reshape(batch size, 句子长度*词典)`
- output: `(batch size, 2)`

```python
#@save
class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT."""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`=句子长度*词典)
        return self.output(X)
```

例子：

We can see that the forward inference of an `NextSentencePred` instance returns binary predictions for each BERT input sequence.

```python
# PyTorch by default won't flatten the tensor as seen in mxnet where, if
# flatten=True, all but the first axis of input data are collapsed together
encoded_X = torch.flatten(encoded_X, start_dim=1)# （batch size, 句子长度*词典）
# input_shape for NSP: (batch size, `num_hiddens`)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

torch.Size([2, 2])
The cross-entropy loss of the 2 binary classifications can also be computed.

```python
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

torch.Size([2])
It is noteworthy that all the labels in both the aforementioned pretraining tasks can be trivially obtained from the pretraining corpus without manual labeling effort. The original BERT has been pretrained on the concatenation of BookCorpus [[Zhu et al., 2015]](https://d2l.ai/chapter_references/zreferences.html#zhu-kiros-zemel-ea-2015) and English Wikipedia. These two text corpora are huge: they have 800 million words and 2.5 billion words, respectively.

## Putting All Things Together

When pretraining BERT, the final loss function is **a linear combination of both** the loss functions for **masked language modeling** **and** **next sentence prediction**. ($Loss_{total} = w1\cdot loss(\text{masked language modeling}) + w2\cdot loss(\text{next sentence prediction})$)

Now we can define the `BERTModel` class by instantiating the three classes `BERTEncoder`, `MaskLM`, and `NextSentencePred`. The forward inference returns the encoded BERT representations `encoded_X`, predictions of masked language modeling `mlm_Y_hat`, and next sentence predictions `nsp_Y_hat`.

```python
#@save
class BERTModel(nn.Module):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## Summary

* Word embedding models such as word2vec and GloVe are context-independent. They assign the same pretrained vector to the same word regardless of the context of the word (if any). It is hard for them to handle well polysemy or complex semantics in natural languages.
* For context-sensitive word representations such as ELMo and GPT, representations of words depend on their contexts.
* ELMo encodes context bidirectionally but uses task-specific architectures (however, it is practically non-trivial to craft a specific architecture for every natural language processing task); while GPT is task-agnostic but encodes context left-to-right.
* BERT combines the best of both worlds: it encodes context bidirectionally and requires minimal architecture changes for a wide range of natural language processing tasks.
* The embeddings of the BERT input sequence are the sum of the token embeddings, segment embeddings, and positional embeddings.
* Pretraining BERT is composed of two tasks: masked language modeling and next sentence prediction. The former is able to encode bidirectional context for representing words, while the latter explicitly models the logical relationship between text pairs.

## Exercises

1. Why does BERT succeed?
2. All other things being equal, will a masked language model require more or fewer pretraining steps to converge than a left-to-right language model? Why?
3. In the original implementation of BERT, the positionwise feed-forward network in `BERTEncoder` (via `d2l.EncoderBlock`) and the fully-connected layer in `MaskLM` both use the Gaussian error linear unit (GELU) :cite:`Hendrycks.Gimpel.2016` as the activation function. Research into the difference between GELU and ReLU.

[Discussions](https://discuss.d2l.ai/t/1490)
