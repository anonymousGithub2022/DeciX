import torch

from utils import *

device = torch.device('cuda')
task_id = 2
exp_res = torch.load('exp_res/subj_' + str(task_id) + '_approach_0.exp')
model, test_loader, vocab_tgt, vocab_src = load_model_data(task_id)

s = "    return u'man {}'.format(command.script[3:])"

x = model.tokenizer.encode(s, return_tensors='pt')
x = x.to(device)
x_len = torch.tensor(len(x[0])).reshape([-1, 1])
model = model.to(device).eval()
out = model(x, x_len)
out = out.max(-1)[1]
print(model.tokenizer.decode(out[0]))