import matplotlib.pyplot as plt
import time

import numpy as np

from utils import *

MAX_TEST_NUM = 100
device = torch.device('cuda')

if not os.path.isdir('res'):
    os.mkdir('res')

def main(task_id):
    model, test_loader, vocab_tgt, vocab_src = load_model_data(task_id)
    # model, test_loader, vocab_tgt, vocab_src = load_model_data(1)
    model = model.to(device)
    model.set_device(device)

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

    exp_1 = MarkovGenExpClass(model, config, device)
    exp_2 = CodeGenExpClass(model, config, device)
    exp_3 = LinearGenExpClass(model, config, device)
    exp_4 = LimeExpClass(model, config, device)
    exp_5 = LemnaExpClass(model, config, device)
    exp_6 = RandomExpClass(model, config, device)

    #
    exp_list = [exp_1, exp_2, exp_3, exp_4, exp_5, exp_6]

    final_deduction_res = [[] for _ in range(len(exp_list))]
    final_augment_res = [[] for _ in range(len(exp_list))]
    final_sys_res = [[] for _ in range(len(exp_list))]

    for i, exp_class in enumerate(exp_list):
        exp_res_list = torch.load('exp_res/subj_' + str(task_id) + '_approach_' + str(i) + '.exp')
        save_path = 'res/subj_' + str(task_id) + '_approach_' + str(i) + '.res'
        for j, exp_res in tqdm(enumerate(exp_res_list)):
            if j >= MAX_TEST_NUM:
                break
            (src_tk, src_len), (out_tk, out_len), explain, overheads = exp_res

            eval_len = out_len + 2
            res_1 = exp_class.eval_exp((src_tk, src_len), (out_tk, out_len), eval_len, explain, 0)
            res_2 = exp_class.eval_exp((src_tk, src_len), (out_tk, out_len), eval_len, explain, 1)
            res_3 = exp_class.eval_exp((src_tk, src_len), (out_tk, out_len), eval_len, explain, 2)

            final_deduction_res[i].append(res_1)
            final_augment_res[i].append(res_2)
            final_sys_res[i].append(res_3)
            save_res = [
                np.concatenate(final_deduction_res[i], axis=1),
                np.concatenate(final_augment_res[i], axis=1),
                np.concatenate(final_sys_res[i], axis=1),
            ]
            torch.save(save_res, save_path)

            # m1 = exp_class.mutation((src_tk, src_len), (out_tk, out_len), explain, 0)
            # m2 = exp_class.mutation((src_tk, src_len), (out_tk, out_len), explain, 1)
            # m3 = exp_class.mutation((src_tk, src_len), (out_tk, out_len), explain, 2)


if __name__ == '__main__':
    seed = 101
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    parser = argparse.ArgumentParser(description='Measure Latency')
    parser.add_argument('--task', default=1, type=int, help='experiment subjects')
    args = parser.parse_args()
    main(args.task)