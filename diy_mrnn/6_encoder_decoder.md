# 6 Encoder-Decoder Architecture

:label:`sec_encoder-decoder`


- “编码器－解码器”架构可以将长度可变的序列作为输入和输出，因此适用于机器翻译等序列转换问题。
- 编码器将长度可变的序列作为输入，并将其转换为具有固定形状的编码状态。
- 解码器将具有固定形状的编码状态映射为长度可变的序列。


As we have discussed in [Section 9.5](https://d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html#sec-machine-translation), machine translation is $\color{red}\text{a major problem}$ domain for sequence transduction models, whose input and output are both $\color{red}\textbf{variable-length sequences}$.

To handle this type of inputs and outputs, we can design an architecture with two major components. -

- The first component is an  *encoder* : it takes a variable-length sequence as the input and transforms it into a state with a fixed shape.
- The second component is a  *decoder* : it maps the encoded state of a fixed shape to a variable-length sequence.

This is called an *encoder-decoder* architecture, which is depicted in [Fig. 9.6.1](https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html#fig-encoder-decoder).


<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/encoder-decoder.svg" width = "50%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 9.6.1 The encoder-decoder architecture.
  	</div>
</center>


Let us take machine translation from English to French as an example. Given an input sequence in English: `“They”, “are”, “watching”, “.”`, this encoder-decoder architecture first encodes the variable-length input into a state, then decodes the state to generate the translated sequence token by token as the output: `“Ils”, “regardent”, “.”`.

Since the `encoder-decoder` architecture forms the basis of different sequence transduction models in subsequent sections, this section will convert this architecture into an interface (接口) that will be implemented later.

## 6.1 (**Encoder**)

In the encoder interface, we just specify that the encoder takes variable-length sequences as the input `X`. The implementation will be provided by any model that inherits this base `Encoder` class.

```python
from torch import nn


#@save
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

## 6.2 [**Decoder**]

In the following decoder interface,

- we add an additional `init_state` function to convert the $\text{\colorbox{black}{encoder output}}$ (`enc_outputs`) into the $\text{\colorbox{black}{encoded state}}$. 编码器的输出 -> 编码后的状态

Note that this step may need extra inputs （可能还需要 额外的输入） such as the valid length of the input, which was explained in [Section 9.5.4](https://d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html#subsec-mt-data-loading). To generate a variable-length sequence token by token, every time the $\color{red}decoder$ may $\text{\colorbox{black}{\color{yellow}map}}$ an input (e.g., the generated token at the previous time step) and the encoded state $\text{\colorbox{black}{\color{yellow}into}}$ an output token at the current time step.

```python
#@save
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

## 6.3 [**Putting the Encoder and Decoder Together**]

In the end, the `encoder-decoder` architecture contains both an `encoder` and a `decoder`, with optionally extra arguments. In the forward propagation,

- the output of the encoder is used to produce the encoded state, and
- this state will be further used by the decoder as one of its input.

```python
#@save
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder: Encoder, decoder: Decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

The term “state” in the encoder-decoder architecture has probably inspired you to implement this architecture using neural networks with states. In the next section, we will see how to apply RNNs to design sequence transduction models based on this encoder-decoder architecture.

## Summary

* The encoder-decoder architecture can handle inputs and outputs that are both variable-length sequences, thus is suitable for sequence transduction problems such as machine translation.
* The encoder takes a variable-length sequence as the input and transforms it into a state with a fixed shape.
* The decoder maps the encoded state of a fixed shape to a variable-length sequence.

## Exercises

1. Suppose that we use neural networks to implement the encoder-decoder architecture. Do the encoder and the decoder have to be the same type of neural network?
2. Besides machine translation, can you think of another application where the encoder-decoder architecture can be applied?

[Discussions](https://discuss.d2l.ai/t/1061)
