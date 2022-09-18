from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader



if __name__=="__main__":
    model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    train_examples = [InputExample(texts=["My first sentence", "My second sentence"], label=0.8),
                      InputExample(texts=["Another pair", "Unrelated sentence"], label=0.3)]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)



    from sentence_transformers import evaluation

    sentences1 = ['This list contains the first column', 'With your sentences', 'You want your model to evaluate on']
    sentences2 = ['Sentences contains the other column', 'The evaluator matches sentences1[i] with sentences2[i]',
                  'Compute the cosine similarity and compares it to scores[i]']
    scores = [0.3, 0.6, 0.2]

    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

    # ... Your other code to load training data

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator,
              evaluation_steps=500)