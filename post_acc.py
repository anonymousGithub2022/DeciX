import os

import numpy as np
import torch
import matplotlib.pyplot as plt

if not os.path.isdir('final_res'):
    os.mkdir('final_res')
color_list = ['r', 'b', 'y', 'g', 'c', 'k']

for task_id in [0, 1, 2]:
    final_res = []
    for approach_id in range(5):
        save_path = 'res/subj_' + str(task_id) + '_approach_' + str(approach_id) + '.res'
        res = torch.load(save_path)
        final_res.append(res)
    exp_1, exp_2, exp_3 = [], [], []
    for i, res in enumerate(final_res):
        plt.plot(res[0].mean(1), color_list[i])
        exp_1.append(res[0].mean(1).reshape([-1, 1]))
    plt.show()

    for i, res in enumerate(final_res):
        plt.plot(res[1].mean(1), color_list[i])
        exp_2.append(res[1].mean(1).reshape([-1, 1]))
    plt.show()

    for i, res in enumerate(final_res):
        plt.plot(res[2].mean(1), color_list[i])
        exp_3.append(res[2].mean(1).reshape([-1, 1]))
    plt.show()
    exp_1 = np.concatenate(exp_1, axis=1)
    # for i in range(len(exp_1)):
    #     exp_1[i] = exp_1[:i + 1].min(0)
    exp_2 = np.concatenate(exp_2, axis=1)
    # for i in range(len(exp_2)):
    #     exp_2[i] = exp_2[:i + 1].max(0)
    exp_3 = np.concatenate(exp_3, axis=1)
    # for i in range(len(exp_3)):
    #     exp_3[i] = exp_3[:i + 1].max(0)
    np.savetxt('final_res/' + str(task_id) + '_' + '1.csv', 100 * exp_1, delimiter=',')
    np.savetxt('final_res/' + str(task_id) + '_' + '2.csv', 100 * exp_2, delimiter=',')
    np.savetxt('final_res/' + str(task_id) + '_' + '3.csv', 100 * exp_3, delimiter=',')


