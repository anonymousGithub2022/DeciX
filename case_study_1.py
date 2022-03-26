import torch

from utils import *


task_id = 1
exp_res = torch.load('exp_res/subj_' + str(task_id) + '_approach_0.exp')
model, test_loader, vocab_tgt, vocab_src = load_model_data(task_id)

src_vocab = {}
for k in vocab_src:
    src_vocab[vocab_src[k]] = k
tgt_vocab = {}
for k in vocab_tgt:
    tgt_vocab[vocab_tgt[k]] = k
test_data = []
for d in test_loader:
    test_data.append(d)

f = open('case_study_1.txt', 'w')
for i, batch in enumerate(exp_res):
    (src_tk, src_len), (out_tk, out_len), explain, overheads = batch
    src_list = [src_vocab[int(d)] + '  ' for i, d in enumerate(src_tk[0][:src_len + 2])]
    src_str = ''.join(src_list)
    tgt_str = [tgt_vocab[int(d)] for d in out_tk[:out_len]]
    exp = [(-d).argsort() for d in explain]
    print(src_str)
    f.write(str(i) + '\n')
    src_str = model.tokenizer.decode(src_tk[0]).replace('<pad>', '')
    f.write(src_str)
    f.write('\n')
    print(''.join(tgt_str))
    f.write('\n')
    tgt_str = model.tokenizer.decode(out_tk).replace('<pad>', '')
    f.write(tgt_str)
    f.write('\n')
    print()
    f.write('\n')

    out_token = model.tokenizer.convert_ids_to_tokens(out_tk[:out_len])
    for i, e in enumerate(exp):
        e = [src_list[d] for d in e[0]]
        print(out_token[i], ':        ', e[:5])
        f.write(out_token[i] + ':        ' + ''.join(e[:5]))
        f.write('\n')
    print('----------------------------------------')
    f.write('----------------------------------------')
    f.write('\n')


