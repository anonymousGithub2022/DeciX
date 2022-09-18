import torch
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso
from sklearn.mixture import GaussianMixture
import numpy as np

from .baseExp import BaseExpClass


class LemnaExp(BaseExpClass):
    def __init__(self, model, config, device):
        super(LemnaExp, self).__init__(model, config, device)
        self.model_num = 4

    def compute_weights(self, mask, target_y):
        weight = []
        output_len = target_y.shape[1]
        for i in range(output_len):
            m_list = [Lasso(fit_intercept=False) for _ in range(self.model_num)]
            new_x = mask
            if (target_y[:, i:i+1] == target_y[0, i]).all():
                weight.append(np.random.random((1, new_x.shape[1])))
            else:
                for m in m_list:
                    m.fit(new_x, target_y[:, i:i + 1].reshape(-1))
                coef = np.concatenate([d.coef_.reshape([1, -1]) for d in m_list])
                weight.append(coef.mean(0))
        weight = [d.reshape([1, -1]) for d in weight]
        return weight

    # def explain(self, x):
    #     src_tk, src_len = x
    #     src_tk, src_len = src_tk.to(self.device), src_len.to(self.device)
    #     mutated_res = self.model.perturb_inputs([src_tk, src_len])
    #
    #     (mutated_x, mutated_len, _) = mutated_res
    #     original_y = self.model(src_tk, src_len)
    #     orig_out_seqs, orig_out_len = self.get_output_seqs(original_y)
    #     max_len = orig_out_len[0] + 2
    #     mutated_y = self.model.predict_batch(mutated_x, mutated_len, max_length=max_len)
    #     mutated_seqs, _ = self.get_output_seqs(mutated_y)
    #     orig_out_len = orig_out_len[0]
    #     orig_out_seqs = orig_out_seqs[0]
    #     target_y = [seq[:orig_out_len].eq(orig_out_seqs[:orig_out_len]).to(torch.float).reshape([1, -1]) for seq in mutated_seqs]
    #     target_y = torch.cat(target_y).detach().cpu().numpy()
    #     weights = self.compute_weights(mutated_x[:, :src_len+1], target_y)
    #
    #     return weights, orig_out_seqs, orig_out_len

    def explain(self, x):
        local_res = self.general_preprocess(x)
        return self.explain_local(local_res)

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
        return weights, orig_y_seqs, orig_y_len



