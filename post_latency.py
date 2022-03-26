import os

import matplotlib.pyplot as plt
import time

import numpy as np

from utils import *

device = torch.device('cuda')

if not os.path.isdir('final_res'):
    os.mkdir('final_res')


def main(task_id):
    final_overheads = []
    for i in range(6):
        exp_res_list = torch.load('exp_res/subj_' + str(task_id) + '_approach_' + str(i) + '.exp')
        overheads = []
        for j, exp_res in tqdm(enumerate(exp_res_list)):
            (src_tk, src_len), (out_tk, out_len), explain, overhead = exp_res
            overheads.append(np.array([i, overhead]).reshape([1, 2]))
        overheads = np.concatenate(overheads).reshape([-1, 2])
        final_overheads.append(overheads)
    final_overheads = np.concatenate(final_overheads, axis=0)
    return final_overheads


if __name__ == '__main__':
    if not os.path.isdir('final_res/overheads'):
        os.mkdir('final_res/overheads')
    for task in range(3):
        res = main(task)
        save_path = 'final_res/overheads/' + str(task) + '.csv'
        np.savetxt(save_path, res, delimiter=',')
