# NLP 流程：

1. pretraining：word2vec, Glove, Subword Embedding, BERT
2. Architecture：MLP, CNN, RNN, Attention (self, non-self)
3. Application：
   1. Sequence Level (single、pair)，只用CLS；
   2. Token level，用多个（所有）token

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://d2l.ai/_images/nlp-map-nli-bert.svg"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 15.7.1 This section feeds pretrained BERT to an MLP-based architecture for natural language inference.
  	</div>
</center>
