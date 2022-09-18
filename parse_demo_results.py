import os.path

import torch

from utils import *

if not os.path.isdir('exp_res'):
    os.mkdir('exp_res')

for task_id in range(3):
    model, test_loader, vocab_tgt, vocab_src = load_model_data(task_id)
    config = {
        "PAD_ID":  model.PAD_ID,
        "SOS_ID":  model.SOS_ID,
        "EOS_ID":  model.EOS_ID,
        "UNK_ID":  model.UNK_ID,
        'sp_token': [],
        'vocab_size': len(vocab_src),
        'src_vocab': vocab_src,
        'tgt_vocab': vocab_tgt,
        'max_length': model.max_length,
        'mutate_num': model.mutate_num,
        'mutate_rate': model.mutate_rate
    }
    device = torch.device('cuda')
    exp_list = [exp(model, config, device) for exp in EXPLAIN_LIST]


    for i, exp_class in enumerate(exp_list):
        approach_name = exp_class.__class__.__name__
        exp_res_list = torch.load('exp_res_new/subj_' + str(task_id) + '_approach_' + approach_name + '.exp')
        print(task_id, i, len(exp_res_list))
        exp_res_list = exp_res_list[:100]
        torch.save(exp_res_list, 'exp_res/subj_' + str(task_id) + '_approach_' + approach_name + '.exp')