import torch

from utils import *


task_id = 2
exp_res = torch.load('exp_res/subj_' + str(task_id) + '_approach_0.exp')

model, test_loader, vocab_tgt, vocab_src = load_model_data(task_id)

src_vocab = {}
for k in vocab_src:
    src_vocab[vocab_src[k]] = k
tgt_vocab = {}
for k in vocab_tgt:
    tgt_vocab[vocab_tgt[k]] = k

for batch in exp_res:
    (src_tk, src_len), (out_tk, out_len), explain, overheads = batch
    src_list = [src_vocab[int(d)] + '  ' for i, d in enumerate(src_tk[0][:src_len + 2])]
    src_str = ''.join(src_list)
    tgt_str = [tgt_vocab[int(d)] for d in out_tk[:out_len]]
    exp = [(-d).argsort() for d in explain]
    print(src_str)
    print(''.join(tgt_str))
    print()

    for i, e in enumerate(exp):
        e = [src_list[d] for d in e[0]]
        print(tgt_str[i], ':        ', e[:5])
    print('----------------------------------------')

