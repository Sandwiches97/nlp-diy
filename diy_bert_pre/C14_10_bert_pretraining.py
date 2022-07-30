import torch
from torch import nn
from d2l_en.pytorch.d2l import torch as d2l
from C14_9_bert_dataset import load_data_wiki
from C14_8_BERT import BERTModel, get_tokens_and_segments
from typing import Tuple, List



def _get_batch_loss_bert(net, loss:nn.CrossEntropyLoss, vocab_size:int, tokens_X,
                         segments_X, valid_lens_X, pred_positions_X,
                         mlm_weights_X, mlm_Y, nsp_Y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_X.reshape(-1), pred_positions_X)
    # Compute MLM Loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute NSP Loss
    nsp_l = loss(nsp_Y_hat, nsp_Y.type(torch.long))
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])

    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')

def get_bert_encoding(net, tokens_a, tokens_b=None):
    """

    :param net:
    :param tokens_a:
    :param tokens_b:
    :return: Bert编码后的 句子
    """
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X



if __name__ == "__main__":
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    net = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2, num_layers=2, dropout=0.2,
                    key_size=128, query_size=128, value_size=128,
                    hid_in_features=128, mlm_in_features=128, nsp_in_features=128)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss()

    train_bert(train_iter, net, loss, len(vocab), devices, 50)

    tokens_a = ['a', 'crane', 'is', 'flying']
    encoded_text = get_bert_encoding(net, tokens_a)
    # 此时的tokens_a变为了：Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
    encoded_text_cls = encoded_text[:, 0, :]
    encoded_text_crane = encoded_text[:, 2, :]
    print(f'{encoded_text.shape}, {encoded_text_cls.shape}, {encoded_text_crane[0][:3]}')

