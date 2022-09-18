import torch
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import numpy as np

from .baseExp import BaseExpClass


class MarkovCausalExp(BaseExpClass):
    def __init__(self, model, config, device):
        super(MarkovCausalExp, self).__init__(model, config, device)
        self.approximate_model = Ridge(alpha=3)

    @staticmethod
    def decompose_dependency(weights, trans_prob):
        # new_weights = [weights[0]]
        # for i in range(1, len(weights)):
        #     w = weights[i] + weights[i - 1] * trans_prob[i]
        #     new_weights.append(w)
        # return new_weights
        new_weights = [weights[0]]
        for i in range(1, len(weights)):
            # w = weights[i][:, :-1]
            w = new_weights[-1] * weights[i][0][-1] + weights[i][:, :-1]
            # w = weights[i] + weights[i - 1] * trans_prob[i]
            new_weights.append(w)
        return new_weights

    def compute_weights(self, mask, target_y):
        weight = []
        output_len = target_y.shape[1]
        for i in range(output_len):
            # m = LinearRegression(fit_intercept=False)
            m = self.approximate_model
            if i == 0:
                new_x = mask
            else:
                new_x = np.concatenate([mask, target_y[:, i - 1:i]], axis=1)
            if (target_y[:, i:i+1] == target_y[0, i]).all():
                weight.append(np.random.random((1, new_x.shape[1])))
            else:
                m.fit(new_x, target_y[:, i:i + 1].reshape(-1))
                weight.append(m.coef_)
        weight = [d.reshape([1, -1]) for d in weight]
        return weight

    def explain(self, x):
        local_res = self.general_preprocess(x)
        return self.explain_local(local_res)

    def get_trans_prob(self, orig_y_seqs, orig_y_len, mutated_y_seqs):
        trans_prob = np.zeros(orig_y_len)
        mutated_seqs = torch.stack(mutated_y_seqs)
        for i in range(orig_y_len - 1):
            prev_num = (mutated_seqs[:, i] == orig_y_seqs[i])
            next_num = (mutated_seqs[:, i + 1] == orig_y_seqs[i + 1])

            join = (prev_num * next_num).sum()
            trans_prob[i + 1] = (join + 1e-12) / (prev_num.sum() + 1e-12)
        return trans_prob

    def explain_local(self, local_res):
        (mask_x, mask_y) = local_res['mask']
        (orig_x_seqs, orig_x_len) = local_res['orig_x']
        (orig_y_seqs, orig_y_len) = local_res['orig_y']
        (mutated_x_seqs, mutated_x_len) = local_res['mutate_x']
        (mutated_y_seqs, mutated_y_len) = local_res['mutate_y']

        weights = self.compute_weights(mask_x, mask_y)
        trans_prob = self.get_trans_prob(orig_y_seqs, orig_y_len, mutated_y_seqs)
        new_weights = self.decompose_dependency(weights, trans_prob)
        return new_weights, orig_y_seqs, orig_y_len


class CausalExp(MarkovCausalExp):
    def __init__(self, model, config, device):
        super(CausalExp, self).__init__(model, config, device)

    def compute_weights(self, mask, target_y):
        weight = []
        output_len = target_y.shape[1]
        for i in range(output_len):
            # m = LinearRegression(fit_intercept=False)
            m = self.approximate_model
            if i == 0:
                new_x = mask
            else:
                new_x = np.concatenate([mask, target_y[:, i - 1:i]], axis=1)

            if (target_y[:, i:i+1] == target_y[0, i]).all():
                weight.append(np.random.random((1, new_x.shape[1])))
            else:
                m.fit(new_x, target_y[:, i:i + 1].reshape(-1))
                weight.append(m.coef_)
        weight = [d.reshape([1, -1]) for d in weight]
        weight = [weight[0]] + [d[:, :-1] for d in weight[1:]]
        return weight

    # def prepare_matrix(self, mask, input_len, original_y, mutated_y):
    #     orig_seqs, orig_len = self.get_output_seqs(original_y)
    #     orig_seqs, orig_len = orig_seqs[0], orig_len[0]
    #     mutated_seqs, mutated_len = self.get_output_seqs(mutated_y)
    #
    #     target_y = [seq[:orig_len].eq(orig_seqs[:orig_len]).to(torch.float).reshape([1, -1]) for seq in mutated_seqs]
    #
    #     target_y = torch.cat(target_y).detach().cpu().numpy()
    #     mask = mask.detach().cpu().numpy()
    #     mask = mask[:, :input_len+1]
    #     return mask, target_y, orig_seqs, orig_len

    # def explain(self, x):
    #     ori_inputs, ori_len = x
    #     assert len(ori_inputs) == 1
    #     batch_size, input_length = ori_inputs.shape
    #
    #     mutated_x = ori_inputs.repeat((self.mutate_num, 1))
    #     mutated_len = ori_len.repeat((self.mutate_num, 1)).to(self.device)
    #     random_x = (torch.rand([self.mutate_num, input_length]) * self.vocab_size).int()
    #     mask = (torch.rand([self.mutate_num, input_length]) < 0.8).float()
    #     mask[:, 0] = 1
    #     mask[:, ori_len + 1:] = 1
    #     random_x * (1 - mask) + mutated_x * mask
    #     mutated_x = random_x * (1 - mask) + mutated_x * mask
    #     mutated_x = mutated_x.to(torch.int64).to(self.device)
    #
    #     # ori_pred_tk, ori_pred_len, ori_pred_logit = self.model.predict_likehood(ori_inputs, ori_len)
    #     x = x[0].to(self.device), x[1].to(self.device)
    #     original_y = self.model(x[0], x[1])
    #     _, orig_len = self.get_output_seqs(original_y)
    #     max_len = orig_len[0] + 2
    #     mutated_y = self.model.predict_batch(mutated_x, mutated_len, max_length=max_len)
    #
    #     mask, target_y, orig_seqs, orig_len = self.prepare_matrix(mask, ori_len, original_y, mutated_y)
    #
    #     weights = self.compute_weights(mask, target_y)
    #     return weights, orig_seqs, orig_len

    def explain_local(self, local_res):
        (mask_x, mask_y) = local_res['mask']
        (orig_x_seqs, orig_x_len) = local_res['orig_x']
        (orig_y_seqs, orig_y_len) = local_res['orig_y']
        (mutated_x_seqs, mutated_x_len) = local_res['mutate_x']
        (mutated_y_seqs, mutated_y_len) = local_res['mutate_y']

        weights = self.compute_weights(mask_x, mask_y)
        return weights, orig_y_seqs, orig_y_len


class MarkovLinearExp(MarkovCausalExp):
    def __init__(self, model, config, device):
        super(MarkovLinearExp, self).__init__(model, config, device)

    def explain_local(self, local_res):
        # _, target_y, orig_out_tk, orig_out_len, trans_prob, mutated_x, src_len = local_res
        (mask_x, mask_y) = local_res['mask']
        (orig_x_seqs, orig_x_len) = local_res['orig_x']
        (orig_y_seqs, orig_y_len) = local_res['orig_y']
        (mutated_x_seqs, mutated_x_len) = local_res['mutate_x']
        (mutated_y_seqs, mutated_y_len) = local_res['mutate_y']

        mutated_x_seqs = mutated_x_seqs.detach().cpu().numpy()[:, :orig_x_len + 1]
        weights = self.compute_weights(mutated_x_seqs, mask_y)
        trans_prob = self.get_trans_prob(orig_y_seqs, orig_y_len, mutated_y_seqs)
        new_weights = self.decompose_dependency(weights, trans_prob)
        return new_weights, orig_y_seqs, orig_y_len
