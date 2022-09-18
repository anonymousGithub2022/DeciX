import os

import numpy as np
import torch
import matplotlib

import matplotlib.pyplot as plt
from utils import EXPLAIN_LIST
from utils import *

if not os.path.isdir('final_res'):
    os.mkdir('final_res')
color_list = ['r', 'b', 'y', 'g', 'c', 'k']

for task_id in [0, 1, 2]:
    final_res = []

    explain_list = [
        DepCausalExp,
        # NoDepCausalExp,
        # DepLinearExp,
        LimeExp,
        LemnaExp,
        RandomExp,
    ]

    approach_name_list = [c.__name__ for c in explain_list]
    for approach_name in approach_name_list:
        save_path = 'res/subj_' + str(task_id) + '_approach_' + approach_name + '.res'
        res = torch.load(save_path)
        final_res.append(res)
    metric_1, metric_2, metric_3 = [], [], []
    s_1, s_2, s_3 = [], [], []
    for i, res in enumerate(final_res):
        output_len = [len(d[0]) for d in res[1]]
        tmp_input_len = [[len(d[1][2][0][0])] * output_len[jjj] for jjj, d in enumerate(res[0])]
        input_len = []
        for d in tmp_input_len:
            input_len.extend(d)

        new_metric = np.concatenate(res[1], axis=1)
        assert len(input_len) == len(new_metric[0])
        frac = [np.where(new_metric[:, d] == 0)[0][0] if len(np.where(new_metric[:, d] == 0)[0]) else  10 for d in range(len(input_len))]
        frac_num = np.array([(f + 1) * 0.1 * l for f, l in zip(frac, input_len)])
        frac_percentage = frac_num / np.array(input_len)
        frac_num = np.average(frac_num)
        frac_percentage = np.average(frac_percentage)
        s_1.append([frac_num, frac_percentage])

        metric = new_metric.mean(1)
        plt.plot(metric, color_list[i])
        metric_1.append(metric.reshape([-1, 1]))
    plt.show()

    for i, res in enumerate(final_res):
        output_len = [len(d[0]) for d in res[2]]
        tmp_input_len = [[len(d[1][2][0][0])] * output_len[jjj] for jjj, d in enumerate(res[0])]
        input_len = []
        for d in tmp_input_len:
            input_len.extend(d)

        new_metric = np.concatenate(res[2], axis= 1)
        assert len(input_len) == len(new_metric[0])
        frac = [np.where(new_metric[:, d] == 1)[0][0] if len(np.where(new_metric[:, d] == 1)[0]) else 10 for d in range(len(input_len))]
        frac_num = np.array([(f + 1) * 0.1 * l for f, l in zip(frac, input_len)])
        frac_percentage = frac_num / np.array(input_len)
        frac_num = np.average(frac_num)
        frac_percentage = np.average(frac_percentage)
        s_2.append([frac_num, frac_percentage])

        metric = new_metric.mean(1)
        plt.plot(metric, color_list[i])
        metric_2.append(metric.reshape([-1, 1]))
    plt.show()

    for i, res in enumerate(final_res):
        output_len = [len(d[0]) for d in res[2]]
        tmp_input_len = [[len(d[1][2][0][0])] * output_len[jjj] for jjj, d in enumerate(res[0])]
        input_len = []
        for d in tmp_input_len:
            input_len.extend(d)

        new_metric = np.concatenate(res[3], axis=1)
        assert len(input_len) == len(new_metric[0])
        frac = frac = [np.where(new_metric[:, d] == 1)[0][0] if len(np.where(new_metric[:, d] == 1)[0]) else 10 for d in range(len(input_len))]
        frac_num = np.array([(f + 1) * 0.1 * l for f, l in zip(frac, input_len)])
        frac_percentage = frac_num / np.array(input_len)
        frac_num = np.average(frac_num)
        frac_percentage = np.average(frac_percentage)
        s_3.append([frac_num, frac_percentage])


        metric = new_metric.mean(1)
        plt.plot(metric, color_list[i])
        metric_3.append(metric.reshape([-1, 1]))
    plt.show()
    metric_1 = np.concatenate(metric_1, axis=1)
    metric_2 = np.concatenate(metric_2, axis=1)
    metric_3 = np.concatenate(metric_3, axis=1)
    # for i in range(len(exp_3)):
    #     exp_3[i] = exp_3[:i + 1].max(0)
    np.savetxt('final_res/' + str(task_id) + '_' + '1.csv', 100 * metric_1, delimiter=',')
    np.savetxt('final_res/' + str(task_id) + '_' + '2.csv', 100 * metric_2, delimiter=',')
    np.savetxt('final_res/' + str(task_id) + '_' + '3.csv', 100 * metric_3, delimiter=',')

    s_1 = np.array(s_1)
    np.savetxt('final_res/new_' + str(task_id) + '_' + '1.csv', s_1, delimiter=',')
    s_2 = np.array(s_2)
    np.savetxt('final_res/new_' + str(task_id) + '_' + '2.csv', s_2, delimiter=',')
    s_3 = np.array(s_3)
    np.savetxt('final_res/new_' + str(task_id) + '_' + '3.csv', s_2, delimiter=',')
