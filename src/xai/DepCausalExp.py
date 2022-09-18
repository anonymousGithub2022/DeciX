import time

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import numpy as np
import copy

from .baseExp import BaseExpClass


class DepCausalExp(BaseExpClass):
    def __init__(self, model, config, device):
        super(DepCausalExp, self).__init__(model, config, device)
        self.approximate_model = Ridge(alpha=1)

    @staticmethod
    def decompose_dependency(weights):
        new_weights = [weights[0]]
        input_len = len(weights[0][0])
        for i in range(1, len(weights)):
            current_w = weights[i]
            w = current_w[:, :input_len]
            for jjj in range(input_len, len(current_w[0])):
                w += new_weights[jjj - input_len] * current_w[0][jjj]
            new_weights.append(w)
        return new_weights

    def compute_weights(self, mask, target_y):
        weight = []
        output_len = target_y.shape[1]
        for i in range(output_len):
            m = copy.deepcopy(self.approximate_model)
            if i == 0:
                new_x = mask
            else:
                new_x = np.concatenate([mask, target_y[:, :i]], axis=1)

            if (target_y[:, i] == target_y[0, i]).all():
                weight.append(np.random.random((1, new_x.shape[1])) * 1e-8)
            else:
                m.fit(new_x, target_y[:, i].reshape(-1))
                weight.append(m.coef_)
        weight = [d.reshape([1, -1]) for d in weight]
        return weight

    def explain(self, x):
        t1 = time.time()
        local_res = self.general_preprocess(x)
        t2 = time.time()
        res = self.explain_local(local_res)
        t3 = time.time()
        print(t2 - t1, t3 - t2)
        return res

    def explain_local(self, local_res):
        (mask_x, mask_y) = local_res['mask']
        (orig_x_seqs, orig_x_len) = local_res['orig_x']
        (orig_y_seqs, orig_y_len) = local_res['orig_y']
        (mutated_x_seqs, mutated_x_len) = local_res['mutate_x']
        (mutated_y_seqs, mutated_y_len) = local_res['mutate_y']

        weights = self.compute_weights(mask_x, mask_y)
        new_weights = self.decompose_dependency(weights)
        return new_weights, orig_y_seqs, orig_y_len


class NoDepCausalExp(DepCausalExp):
    def __init__(self, model, config, device):
        super(NoDepCausalExp, self).__init__(model, config, device)

    def compute_weights(self, mask, target_y):
        weight = []
        output_len = target_y.shape[1]
        for i in range(output_len):
            m = copy.deepcopy(self.approximate_model)
            if i == 0:
                new_x = mask
            else:
                new_x = np.concatenate([mask, target_y[:, i - 1:i]], axis=1)

            if (target_y[:, i] == target_y[0, i]).all():
                weight.append(np.random.random((1, new_x.shape[1])))
            else:
                m.fit(new_x, target_y[:, i].reshape(-1))
                weight.append(m.coef_)
        weight = [d.reshape([1, -1]) for d in weight]
        input_len = len(weight[0][0])
        weight = [weight[0]] + [d[:, :input_len] for d in weight[1:]]
        return weight

    def explain_local(self, local_res):
        (mask_x, mask_y) = local_res['mask']
        (orig_x_seqs, orig_x_len) = local_res['orig_x']
        (orig_y_seqs, orig_y_len) = local_res['orig_y']
        (mutated_x_seqs, mutated_x_len) = local_res['mutate_x']
        (mutated_y_seqs, mutated_y_len) = local_res['mutate_y']

        weights = self.compute_weights(mask_x, mask_y)
        return weights, orig_y_seqs, orig_y_len


class DepLinearExp(DepCausalExp):
    def __init__(self, model, config, device):
        super(DepLinearExp, self).__init__(model, config, device)

    def explain_local(self, local_res):
        (mask_x, mask_y) = local_res['mask']
        (orig_x_seqs, orig_x_len) = local_res['orig_x']
        (orig_y_seqs, orig_y_len) = local_res['orig_y']
        (mutated_x_seqs, mutated_x_len) = local_res['mutate_x']
        (mutated_y_seqs, mutated_y_len) = local_res['mutate_y']

        if self.task == 'DeepAPI':
            mutated_x_seqs = mutated_x_seqs.detach().cpu().numpy()[:, :orig_x_len + 2]
        else:
            mutated_x_seqs = mutated_x_seqs.detach().cpu().numpy()[:, :orig_x_len]

        weights = self.compute_weights(mutated_x_seqs, mask_y)
        new_weights = self.decompose_dependency(weights)
        return new_weights, orig_y_seqs, orig_y_len
