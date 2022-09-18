import os
import time

import matplotlib.pyplot as plt

from utils import *

if not os.path.isdir('exp_res'):
    os.mkdir('exp_res')

device = torch.device('cuda')
IS_EVALUATION = False
MAX_TEST_NUM = 20 if IS_EVALUATION else 1000


def evaluation(jjj, exp, explain, src_tk, src_len, out_tk, out_len):
    t1 = time.time()
    eval_len = out_len + 2
    res_1 = exp.eval_exp((src_tk, src_len), (out_tk, out_len), eval_len, explain, 0)
    t2 = time.time()
    res_2 = exp.eval_exp((src_tk, src_len), (out_tk, out_len), eval_len, explain, 1)
    t3 = time.time()
    res_3 = exp.eval_exp((src_tk, src_len), (out_tk, out_len), eval_len, explain, 2)
    t4 = time.time()
    print(jjj, 'eval', t2 - t1, t3 - t2, t4 - t3)
    return res_1, res_2, res_3


def main(task_id):
    model, test_loader, vocab_tgt, vocab_src = load_model_data(task_id)
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

    exp_list = [exp(model, config, device) for exp in EXPLAIN_LIST]
    base_exp = BaseExpClass(model, config, device)
    exp_res = [[] for _ in range(len(exp_list))]
    final_deduction_res = [[] for _ in range(len(exp_list))]
    final_augment_res = [[] for _ in range(len(exp_list))]
    final_sys_res = [[] for _ in range(len(exp_list))]

    for i, batch in tqdm(enumerate(test_loader)):
        if i > MAX_TEST_NUM:
            break
        if task_id == 0:
            src_tk, src_len, tgt_tk, tgt_len = batch  # deepAPI
        elif task_id == 1:
            src_tk, src_len = batch[0], batch[1].sum(1)
        elif task_id == 2:
            src_tk, src_len = batch, torch.tensor(len(batch[0])).reshape([-1, 1])
        else:
            raise NotImplementedError
        local_res = base_exp.general_preprocess([src_tk, src_len])
        for jjj, exp in enumerate(exp_list):
            # try:
            t1 = time.time()
            if hasattr(exp, 'explain_local'):
                explain, out_tk, out_len = exp.explain_local(local_res)
            else:
                explain, out_tk, out_len = exp.explain([src_tk, src_len])
            t2 = time.time()
            assert type(out_len) == int
            assert len(explain) == out_len

            if exp.task == 'DeepAPI':
                assert len(explain[0][0]) == int(src_len) + 2
            elif exp.task == 'CodeBert':
                assert len(explain[0][0]) == int(src_len)
            elif exp.task == 'GPT2':
                assert len(explain[0][0]) == int(src_len)
            res = [
                (src_tk.detach().cpu(), src_len.detach().cpu()),
                (out_tk.detach().cpu(), out_len),
                explain, t2 - t1
            ]
            print(i, jjj, 'success', t2 - t1)
            exp_res[jjj].append(res)

            approach_name = exp.__class__.__name__
            torch.save(exp_res[jjj], 'exp_res/subj_' + str(task_id) + '_approach_' + approach_name + '.exp')

            # except Exception as e:
            #     print('ERROR', i, jjj, e)
            #     continue

            if IS_EVALUATION:
                res_1, res_2, res_3 = evaluation(jjj, exp, explain, src_tk, src_len, out_tk, out_len)
                final_deduction_res[jjj].append(res_1)
                final_augment_res[jjj].append(res_2)
                final_sys_res[jjj].append(res_3)

    color_list = ['r', 'b', 'g', 'y', 'c', 'w', 'm']
    if IS_EVALUATION:
        final_deduction_res = [np.concatenate(d, axis=1).mean(1) for d in final_deduction_res]
        for color_id, data in enumerate(final_deduction_res):
            plt.plot(data, color_list[color_id])
        plt.show()

        final_augment_res = [np.concatenate(d, axis=1).mean(1) for d in final_augment_res]
        for color_id, data in enumerate(final_augment_res):
            plt.plot(data, color_list[color_id])
        plt.show()

        final_sys_res = [np.concatenate(d, axis=1).mean(1) for d in final_sys_res]
        for color_id, data in enumerate(final_sys_res):
            plt.plot(data, color_list[color_id])
        plt.show()


if __name__ == '__main__':
    seed = 101
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    parser = argparse.ArgumentParser(description='Measure Latency')
    parser.add_argument('--task', default=0, type=int, help='experiment subjects')
    args = parser.parse_args()
    main(args.task)
