import torch
import torch.nn as nn


class Wrapper_Base(nn.Module):
    def __init__(self, model, src_vocab_size):
        super(Wrapper_Base, self).__init__()
        self.model = model
        self.device = None
        self.max_length = self.model.max_length
        self.batch_num = 50
        self.mutate_num = 500
        self.mutate_rate = 0.5
        self.src_vocab_size = src_vocab_size
        # self.tokenizer = model.tokenizer

    def set_device(self, device):
        self.device = device

    def bernoulli_perturb(self, x, **kwargs):
        ori_inputs, ori_len = x
        batch_size, input_length = ori_inputs.shape

        mutated_x = ori_inputs.clone()
        mutated_len = ori_len.clone()
        random_x = (torch.rand([1, input_length]) * self.src_vocab_size).int().to(self.device)
        mask = (torch.rand([1, input_length], device=self.device) < self.mutate_rate).float()
        mask[:, 0] = 1
        mask[:, ori_len + 1:] = 1
        mutated_x = random_x * (1 - mask) + mutated_x * mask
        mutated_x = mutated_x.to(torch.int64).to(self.device)
        return mutated_x, mutated_len

    def predict_batch(self, src_tks, src_lens, max_length=None, index=None):
        if max_length is None:
            max_length = self.max_length
        iter_num = len(src_tks) // self.batch_num
        if iter_num * self.batch_num != len(src_tks):
            iter_num = iter_num + 1
        mutated_y = []
        for i in range(iter_num):
            st, ed = self.batch_num * i, min(len(src_tks), self.batch_num * (i + 1))
            y = self(src_tks[st:ed], src_lens[st:ed], max_length)
            mutated_y.append(y.detach().cpu())
        mutated_y = torch.cat(mutated_y)
        return mutated_y

    def format_input_baseline(self, x, oov_tk):
        ori_inputs, ori_len = x
        baseline = torch.ones_like(ori_inputs) * oov_tk
        baseline[:, 0] = ori_inputs[:, 0]
        baseline[:, ori_len + 1] = ori_inputs[:, ori_len + 1]
        return baseline, ori_len


class Wrapper_DeepAPI(Wrapper_Base):
    def __init__(self, model, src_vocab_size):
        super(Wrapper_DeepAPI, self).__init__(model, src_vocab_size)
        self.PAD_ID = 0
        self.SOS_ID = 1
        self.EOS_ID = 2
        self.UNK_ID = 3
        self.batch_num = 250
        self.task = 'DeepAPI'

    def get_embedding(self, src_seqs, src_lens):
        return self.model.get_embedding(src_seqs, src_lens)

    def forward(self, src_seqs, src_lens, max_length=None, index=None):
        src_seqs, src_lens = src_seqs.to(self.device), src_lens.to(self.device)
        return self.model(src_seqs, src_lens, max_length, index)

    def perturb_inputs(self, x, **kwargs):
        ori_inputs, ori_len = x
        assert len(ori_inputs) == 1
        batch_size, input_length = ori_inputs.shape

        mutated_x = ori_inputs.repeat((self.mutate_num, 1)).to(self.device)
        mutated_len = ori_len.repeat((self.mutate_num, 1)).to(self.device)
        random_x = (torch.rand([self.mutate_num, input_length], device=self.device) * self.src_vocab_size).int()
        mask = (torch.rand([self.mutate_num, input_length], device=self.device) < self.mutate_rate).float()
        mask[:, 0] = 1
        mask[:, ori_len + 1:] = 1
        mutated_x = random_x * (1 - mask) + mutated_x * mask
        mutated_x = mutated_x.to(torch.int64).to(self.device)
        return mutated_x, mutated_len, mask


class Wrapper_CodeBert(Wrapper_Base):
    def __init__(self, model, src_vocab_size):
        super(Wrapper_CodeBert, self).__init__(model, src_vocab_size)
        self.vocab = self.model.lm_head.out_features
        self.model.max_length = 100
        self.PAD_ID = 1
        self.SOS_ID = 0
        self.EOS_ID = 2
        self.UNK_ID = 3
        self.max_length = self.model.max_length
        self.tokenizer = model.tokenizer
        self.task = 'CodeBert'
        self.batch_num = 100

    def get_embedding(self, src_seqs, src_lens):
        mask = torch.zeros_like(src_seqs)
        for i in range(len(src_lens)):
            mask[i, :src_lens[i]] = 1
        return self.model.encoder(src_seqs, attention_mask=mask).last_hidden_state

    @torch.no_grad()
    def forward(self, src_seqs, src_lens, max_length=None, index=None):
        src_seqs, src_lens = src_seqs.to(self.device), src_lens.to(self.device)
        if max_length == None:
            max_length = self.model.max_length
        mask = torch.zeros_like(src_seqs)
        for i in range(len(src_lens)):
            mask[i, :src_lens[i]] = 1
        if self.model.beam_size == 1:
            prediction = self.model.predict_likehood_batch(src_seqs, mask, max_length)
        else:
            prediction = self.model.predict_likehood(src_seqs, mask)
        predict_tk = prediction.detach().cpu()
        predict_tk = predict_tk[:, :self.max_length]
        predict_likehood = nn.functional.one_hot(predict_tk, num_classes=self.vocab) #TODO
        assert (predict_likehood.max(-1)[1] == predict_tk).all()

        return predict_likehood

    def perturb_inputs(self, x, **kwargs):
        ori_inputs, ori_len = x
        assert len(ori_inputs) == 1
        batch_size, input_length = ori_inputs.shape

        mutated_x = ori_inputs.repeat((self.mutate_num, 1)).to(self.device)
        mutated_len = ori_len.repeat((self.mutate_num, 1)).to(self.device)
        random_x = (torch.rand([self.mutate_num, input_length], device=self.device) * self.src_vocab_size).int()
        mask = (torch.rand([self.mutate_num, input_length], device=self.device) < self.mutate_rate).float()
        mask[:, 0] = 1
        mask[:, ori_len - 1:] = 1
        mutated_x = random_x * (1 - mask) + mutated_x * mask
        mutated_x = mutated_x.to(torch.int64).to(self.device)
        return mutated_x, mutated_len, mask


class Wrapper_GPT2(Wrapper_Base):
    def __init__(self, model, src_vocab_size):
        super(Wrapper_GPT2, self).__init__(model, src_vocab_size)
        self.vocab = self.model.lm_head.out_features
        self.PAD_ID = 0
        self.SOS_ID = None
        self.EOS_ID = self.vocab
        self.UNK_ID = None
        self.model.max_length = 100  # model.config.max_length
        self.max_length = 100
        self.task = 'GPT2'
        self.batch_num = 100

    def get_embedding(self, src_seqs, src_lens):
        vec = self.model.transformer.wte(src_seqs)
        return vec

    @torch.no_grad()
    def forward(self, src_seqs, src_lens, max_length=None, index=None):
        src_seqs = src_seqs.to(self.device)
        if max_length == None:
            max_length = self.model.max_length
        prediction = self.model.generate(src_seqs, max_length=max_length)
        predict_tk = prediction.detach().cpu()
        predict_tk = predict_tk[:, :self.max_length]
        predict_likehood = nn.functional.one_hot(predict_tk, num_classes=self.vocab)  # TODO
        assert (predict_likehood.max(-1)[1] == predict_tk).all()
        return predict_likehood

    def perturb_inputs(self, x, **kwargs):
        ori_inputs, ori_len = x
        assert len(ori_inputs) == 1
        batch_size, input_length = ori_inputs.shape

        mutated_x = ori_inputs.repeat((self.mutate_num, 1)).to(self.device)
        mutated_len = ori_len.repeat((self.mutate_num, 1)).to(self.device)
        random_x = (torch.rand([self.mutate_num, input_length], device=self.device) * self.src_vocab_size).int()
        mask = (torch.rand([self.mutate_num, input_length], device=self.device) < self.mutate_rate).float()
        mutated_x = random_x * (1 - mask) + mutated_x * mask
        mutated_x = mutated_x.to(torch.int64).to(self.device)
        return mutated_x, mutated_len, mask

    def bernoulli_perturb(self, x, **kwargs):
        ori_inputs, ori_len = x
        batch_size, input_length = ori_inputs.shape

        mutated_x = ori_inputs.clone()
        mutated_len = ori_len.clone()
        random_x = (torch.rand([1, input_length]) * self.src_vocab_size).int().to(self.device)
        mask = (torch.rand([1, input_length], device=self.device) < self.mutate_rate).float()
        mutated_x = random_x * (1 - mask) + mutated_x * mask
        mutated_x = mutated_x.to(torch.int64).to(self.device)
        return mutated_x, mutated_len









# class Wrapper_Code2Seq(Wrapper_Base):
#     def __init__(self, model, src_vocab_size):
#         super(Wrapper_Code2Seq, self).__init__(model, src_vocab_size)
#
#     def forward(self, src_seqs, src_lens, max_length=None, index=None):
#         from_token, path_nodes, to_token = src_seqs
#         from_token, path_nodes, to_token = \
#             from_token.to(self.device), path_nodes.to(self.device), to_token.to(self.device)
#         prediction = self.model(from_token, path_nodes, to_token, src_lens)
#
#         return prediction
#
#     def perturb_inputs(self, x, **kwargs):
#         ori_inputs, ori_len = x
#         from_token, path_nodes, to_token = ori_inputs
#         assert len(ori_len) == 1
#
#         mutated_from_token = from_token.repeat((self.mutate_num, 1)).to(self.device)
#         mutated_to_token = to_token.repeat((self.mutate_num, 1)).to(self.device)
#         mutated_path_nodes = path_nodes.repeat((self.mutate_num, 1)).to(self.device)
#         mutated_len = ori_len.repeat((self.mutate_num)).to(self.device)
#
#         random_from = (torch.rand(from_token.shape, device=self.device) * self.src_vocab_size).int()
#         mask_from = (torch.rand(from_token.shape, device=self.device) < self.mutate_rate).float()
#
#         for i in range(int(ori_len)):
#             print()
#             column_len = len(from_token[:, i]) - sum(from_token[:, i] == 0)
#             column_index = [i + j * int(ori_len) for j in range(self.mutate_num)]
#             mask_from[column_len:, column_index] = 1
#
#         mutated_from = random_from * (1 - mask_from) + mutated_from_token * mask_from
#         mutated_from = mutated_from.to(torch.int64).to(self.device)
#         return (mutated_from, mutated_path_nodes, mutated_to_token), mutated_len, mask_from
#
#     def bernoulli_perturb(self, x, **kwargs):
#         print()
#         ori_inputs, ori_len = x
#         batch_size, input_length = ori_inputs.shape
#
#         mutated_x = ori_inputs.clone()
#         mutated_len = ori_len.clone()
#         random_x = (torch.rand([1, input_length]) * self.src_vocab_size).int().to(self.device)
#         mask = (torch.rand([1, input_length], device=self.device) < self.mutate_rate).float()
#         mask[:, 0] = 1
#         mask[:, ori_len + 1:] = 1
#         mutated_x = random_x * (1 - mask) + mutated_x * mask
#         mutated_x = mutated_x.to(torch.int64).to(self.device)
#         return mutated_x, mutated_len
#
#     def predict_batch(self, src_tks, src_lens, max_length=None, index=None):
#         print()
#         if max_length is None:
#             max_length = self.max_length
#         iter_num = len(src_tks) // self.batch_num
#         if iter_num * self.batch_num != len(src_tks):
#             iter_num = iter_num + 1
#         mutated_y = []
#         for i in range(iter_num):
#             st, ed = self.batch_num * i, min(len(src_tks), self.batch_num * (i + 1))
#             y = self(src_tks[st:ed], src_lens[st:ed], max_length)
#             mutated_y.append(y)
#         mutated_y = torch.cat(mutated_y)
#         return mutated_y
