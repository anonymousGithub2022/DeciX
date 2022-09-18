import torch
import numpy as np


class BaseExpClass:
    def __init__(self, model, config, device):
        model.to(device)
        self.model = model
        self.config = config
        self.device = device
        self.task = model.task

        self.vocab_size = config['vocab_size']
        self.sp_token = config['sp_token']

        self.PAD_ID = config['PAD_ID']
        self.SOS_ID = config['SOS_ID']
        self.EOS_ID = config['EOS_ID']
        self.UNK_ID = config['UNK_ID']

        self.src_vocab = config['src_vocab']
        self.tgt_vocab = config['tgt_vocab']

        self.mutate_num = config['mutate_num']
        self.mutate_rate = config['mutate_rate']
        self.max_length = config['max_length']

        self.DEDUCTION = 0
        self.AUGMENT = 1
        self.SYNTHETHIC = 2

        self.batch_num = 250

    def prepare_matrix(self, mask_x, input_len, original_y, mutated_y_seqs):
        orig_y_seqs, orig_y_len = self.get_output_seqs(original_y)
        orig_y_seqs, orig_y_len = orig_y_seqs[0], orig_y_len[0]

        mask_y = [
            seq[:orig_y_len].eq(orig_y_seqs[:orig_y_len]).to(torch.float).reshape([1, -1])
            for seq in mutated_y_seqs
        ]
        mask_y = torch.cat(mask_y).detach().cpu().numpy()

        mask_x = mask_x.detach().cpu().numpy()
        if self.task == 'DeepAPI':
            mask_x = mask_x[:, :input_len + 2]
        else:
            mask_x = mask_x[:, :input_len]

        # return mask_x, mask_y, orig_seqs, orig_len, trans_prob
        return mask_x, mask_y, orig_y_seqs, orig_y_len

    def general_preprocess(self, x):
        orig_x_seqs, orig_x_len = x
        original_y = self.model(orig_x_seqs, orig_x_len)

        mutated_res = self.model.perturb_inputs([orig_x_seqs, orig_x_len])
        (mutated_x_seqs, mutated_x_len, mask_x) = mutated_res

        _, orig_len = self.get_output_seqs(original_y)

        mutated_y_seqs, mutated_y_len = self.predict_output_seqs(mutated_x_seqs, mutated_x_len)
        tmp_res = self.prepare_matrix(mask_x, orig_x_len, original_y, mutated_y_seqs)

        mask_x, mask_y, orig_y_seqs, orig_y_len = tmp_res
        res = {
            'mask': (mask_x, mask_y),
            'orig_x': (orig_x_seqs, orig_x_len),
            'orig_y': (orig_y_seqs, orig_y_len),
            'mutate_x': (mutated_x_seqs, mutated_x_len),
            'mutate_y': (mutated_y_seqs, mutated_y_len)
        }
        return res

    def remove_pad(self, seq):
        out_len = len(seq)
        for i, s in enumerate(seq):
            if s == self.EOS_ID:
                seq[i+1:] = self.PAD_ID
                out_len = i + 1
        return seq, out_len

    def get_output_seqs(self, prob):
        out_seqs = prob.max(-1)[1]
        out_seqs = [self.remove_pad(s) for s in out_seqs]

        out_len = [s[1] for s in out_seqs]
        out_seqs = [s[0].to(self.device) for s in out_seqs]
        return out_seqs, out_len

    @torch.no_grad()
    def predict_output_seqs(self, mutated_x, mutated_len, max_len=None):
        mutated_x = mutated_x.to(self.device)
        mutated_len = mutated_len.to(self.device)
        mutated_y = self.model.predict_batch(mutated_x, mutated_len, max_length=max_len)
        return self.get_output_seqs(mutated_y)

    def explain(self, x):
        pass

    @staticmethod
    def compare_with_orig(out_seqs, y, index):
        tmp = [seq[index].eq(y[0][index]).to(torch.float).reshape([1, -1]) for seq in out_seqs]
        return torch.cat(tmp).detach().cpu().numpy()

    def eval_exp(self, x, y, max_len, exp_list, eval_type):
        test_inputs, NUM = self.mutation(x, y, exp_list, eval_type)
        res = self._eval_mutants(x, y, max_len, exp_list, test_inputs, NUM)
        return res

    def _eval_mutants(self,  x, y, max_len, exp_list, test_inputs, NUM):
        tmp_x_len = x[1].repeat((len(test_inputs), 1))
        out_seqs, out_len = self.predict_output_seqs(test_inputs, tmp_x_len, max_len)

        out_seqs = [out_seqs[i * NUM:(i + 1) * NUM] for i in range(len(exp_list))]

        res = [self.compare_with_orig(out_s, y, i) for i, out_s in enumerate(out_seqs)]
        return np.concatenate(res, axis=1)

    def mutation(self, x, y, exp_list, eval_type):
        assert y[1] == len(exp_list)
        if self.task == 'DeepAPI':
            mask_list = [int((x[1] + 2) * jjj * 0.1) for jjj in range(1, 11)]
        else:
            mask_list = [int(x[1] * jjj * 0.1) for jjj in range(1, 11)]
        NUM = len(mask_list)
        test_inputs = []
        for i, exp in enumerate(exp_list):
            import_index = (-1 * exp).argsort()[0]
            if eval_type == self.DEDUCTION:
                tmp_x = self.deduction_mutation(x, import_index, mask_list)
            elif eval_type == self.AUGMENT:
                tmp_x = self.augment_mutation(x, import_index, mask_list)
            elif eval_type == self.SYNTHETHIC:
                tmp_x = self.synthetic_mutation(x, import_index, mask_list)
            else:
                raise NotImplementedError
            test_inputs.append(tmp_x)
        test_inputs = torch.cat(test_inputs)
        return test_inputs, NUM

    def deduction_mutation(self, x, import_index, mask_list):
        tmp_x = x[0].repeat((len(mask_list), 1))
        for j, mask_num in enumerate(mask_list):
            tmp_x[j, import_index[:mask_num]] = self.PAD_ID
            # tmp_x[j, import_index[:mask_num]] = int(np.random.rand() * self.vocab_size)
        return tmp_x

    def augment_mutation(self, x, import_index, mask_list):
        tmp_x = torch.ones([len(mask_list), x[0].shape[1]], device=x[0].device, dtype=x[0].dtype) * self.PAD_ID
        if self.task == 'DeepAPI':
            tmp_x[:, 0] = 1
            tmp_x[:, int(x[1]) + 1] = 2
            assert x[0][0, int(x[1]) + 1] == 2
            assert x[0][0, 0] == 1
        elif self.task == 'CodeBert':
            tmp_x[:, 0] = 0
            tmp_x[:, int(x[1]) - 1] = 2
            assert x[0][0, int(x[1]) - 1] == 2
            assert x[0][0, 0] == 0
        elif self.task == 'GPT2':
            pass
        else:
            raise NotImplementedError
        for j, mask_num in enumerate(mask_list):
            tmp_x[j, import_index[:mask_num]] = x[0][0, import_index[:mask_num]]
        return tmp_x

    def synthetic_mutation(self, x, import_index, mask_list):
        tmp_x = (torch.rand([len(mask_list), x[0].shape[1]], device=x[0].device) * self.vocab_size).to(x[0].dtype)
        if self.task == 'DeepAPI':
            tmp_x[:, 0] = 1
            tmp_x[:, int(x[1]) + 1] = 2
            tmp_x[:, (int(x[1])+2):] = self.PAD_ID
        elif self.task == 'CodeBert':
            tmp_x[:, 0] = 0
            tmp_x[:, int(x[1]) - 1] = 2
            tmp_x[:, int(x[1]):] = self.PAD_ID
        elif self.task == 'GPT2':
            pass
        else:
            raise NotImplementedError
        for j, mask_num in enumerate(mask_list):
            tmp_x[j, import_index[:mask_num]] = x[0][0, import_index[:mask_num]]
        return tmp_x

    # def perturb_inputs(self, x, **kwargs):
    #     ori_inputs, ori_len = x
    #     assert len(ori_inputs) == 1
    #     batch_size, input_length = ori_inputs.shape
    #
    #     mutated_x = ori_inputs.repeat((self.mutate_num, 1)).to(self.device)
    #     mutated_len = ori_len.repeat((self.mutate_num, 1)).to(self.device)
    #     random_x = (torch.rand([self.mutate_num, input_length], device=self.device) * self.vocab_size).int()
    #     mask = (torch.rand([self.mutate_num, input_length], device=self.device) < self.mutate_rate).float()
    #     mask[:, 0] = 1
    #     mask[:, ori_len + 1:] = 1
    #     mutated_x = random_x * (1 - mask) + mutated_x * mask
    #     mutated_x = mutated_x.to(torch.int64).to(self.device)
    #     return mutated_x, mutated_len, mask


class RandomExp(BaseExpClass):
    def __init__(self, model, config, device):
        super(RandomExp, self).__init__(model, config, device)

    def explain(self, x):
        src_tk, src_len = x
        src_tk, src_len = src_tk.to(self.device), src_len.to(self.device)
        original_y = self.model(src_tk, src_len)
        orig_out_seqs, orig_out_len = self.get_output_seqs(original_y)

        if self.task == 'DeepAPI':
            weights = [np.random.random([1, src_len + 2]) for _ in range(orig_out_len[0])]
        else:
            weights = [np.random.random([1, src_len]) for _ in range(orig_out_len[0])]
        return weights, orig_out_seqs[0], orig_out_len[0]


    # def eval_exp(self, x, y, max_len, exp_list, eval_type):
    #     assert y[1] == len(exp_list)
    #     mask_list = [int(x[1] * jjj * 0.1) for jjj in range(1, 10)]
    #     final_res = []
    #     for i, exp in enumerate(exp_list):
    #         import_index = (-1 * exp).argsort()[0]
    #         tmp_x_len = x[1].repeat((len(mask_list), 1))
    #         if eval_type == self.DEDUCTION:
    #             tmp_x = self.deduction_mutation(x, import_index, mask_list)
    #         elif eval_type == self.AUGMENT:
    #             tmp_x = self.augment_mutation(x, import_index, mask_list)
    #         elif eval_type == self.SYNTHETHIC:
    #             tmp_x = self.synthetic_mutation(x, import_index, mask_list)
    #         else:
    #             raise NotImplementedError
    #         out_seqs, out_len = self.predict_output_seqs(tmp_x, tmp_x_len, max_len)
    #         res = [seq[i].eq(y[0][i]).to(torch.float).reshape([1, -1]) for seq in out_seqs]
    #         res = torch.cat(res).detach().cpu().numpy()
    #         final_res.append(res.reshape([-1, 1]))
    #     final_res = np.concatenate(final_res, axis=1)
    #     return final_res