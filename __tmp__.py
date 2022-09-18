import time

import matplotlib.pyplot as plt

from utils import *

TASK_ID = 2


def main(seed=101):
    # Set the random seed manually for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model, test_loader, vocab_tgt, vocab_src = load_model_data(TASK_ID)
    # model, test_loader, vocab_tgt, vocab_src = load_model_data(1)
    device = torch.device(6)
    model = model.to(device)
    model.set_device(device)

    config = {
        "PAD_ID":  0,
        "SOS_ID": 1,
        "EOS_ID": 2,
        "UNK_ID": 3,
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

    exp_4 = LimeExpClass(model, config, device)
    exp_3 = LemnaExpClass(model, config, device)
    exp_5 = LemnaExpClass(model, config, device)


    #
    exp_list = [exp_1, exp_5, exp_2, exp_3, exp_4]
    final_deduction_res = [[] for _ in range(len(exp_list))]
    final_augment_res = [[] for _ in range(len(exp_list))]
    final_sys_res = [[] for _ in range(len(exp_list))]
    for i, batch in tqdm(enumerate(test_loader)):
        if i > 100:
            continue
        for jjj, exp in enumerate(exp_list):
            if TASK_ID == 0:
                src_tk, src_len, tgt_tk, tgt_len = batch # deepAPI
            elif TASK_ID == 1:
                src_tk, src_len = batch[0], batch[1].sum(1)
            elif TASK_ID == 2:
                src_tk, src_len = batch, torch.tensor(len(batch[0])).reshape([-1, 1])
            else:
                raise NotImplementedError

            t1 = time.time()
            explain, out_tk, out_len = exp.explain([src_tk, src_len])
            t2 = time.time()
            assert len(explain) == out_len
            print(jjj, 'explain', t2 - t1)

            t1 = time.time()
            eval_len = out_len + 2
            res_1 = exp.eval_exp((src_tk, src_len), (out_tk, out_len), eval_len, explain, 0)
            t2 = time.time()
            res_2 = exp.eval_exp((src_tk, src_len), (out_tk, out_len), eval_len, explain, 1)
            t3 = time.time()
            res_3 = exp.eval_exp((src_tk, src_len), (out_tk, out_len), eval_len, explain, 2)
            t4 = time.time()
            print(jjj, 'eval', t2 - t1, t3 - t2, t4 - t3)

            final_deduction_res[jjj].append(res_1)
            final_augment_res[jjj].append(res_2)
            final_sys_res[jjj].append(res_3)

    final_deduction_res = [np.concatenate(d, axis=1).mean(1) for d in final_deduction_res]
    plt.plot(final_deduction_res[0], 'r')
    plt.plot(final_deduction_res[1], 'b')
    plt.plot(final_deduction_res[2], 'g')
    plt.plot(final_deduction_res[3], 'y')
    plt.show()

    final_augment_res = [np.concatenate(d, axis=1).mean(1) for d in final_augment_res]
    plt.plot(final_augment_res[0], 'r')
    plt.plot(final_augment_res[1], 'b')
    plt.plot(final_augment_res[2], 'g')
    plt.plot(final_augment_res[3], 'y')
    plt.show()

    final_sys_res = [np.concatenate(d, axis=1).mean(1) for d in final_sys_res]
    plt.plot(final_sys_res[0], 'r')
    plt.plot(final_sys_res[1], 'b')

    plt.plot(final_sys_res[2], 'g')
    plt.plot(final_sys_res[3], 'y')
    plt.show()

    # metrics = Metrics()
    # test_deepAPI(model, metrics, test_loader, vocab_desc, vocab_api, n_samples, decode_mode, f_eval)
    # f_eval.close()


if __name__ == '__main__':

    main()