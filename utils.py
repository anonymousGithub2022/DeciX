import os
from src import *

if not os.path.isdir('./results'):
    os.mkdir('./results')

TASK_LIST = [
    'deepAPI',
    'CodeBert',
    'GPT2'
]
for take_name in TASK_LIST:
    sub_dir = './results/' + take_name
    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)


def _load_deepAPI():
    data_dir = '/home/sxc180080/data/Project/CodeGenExp/data/deepAPI/'
    model_path = '/home/sxc180080/data/Project/CodeGenExp/model_weight/deepAPI/model_epo120000.pkl'
    model, test_loader, vocab_api, vocab_desc =\
        load_deepAPI_model_data(data_dir, model_path)
    model = Wrapper_DeepAPI(model, len(vocab_desc))
    return model, test_loader, vocab_api, vocab_desc


def _load_CodeBert():
    model_path = '/home/sxc180080/data/Project/CodeGenExp/src/CodeBert/roberta/checkpoint-best-bleu/pytorch_model.bin'
    data_path = ''.join([
        '/home/sxc180080/data/Project/Dataset/CodeXGlue/code-to-code/test.java-cs.txt.java'
        ','
        '/home/sxc180080/data/Project/Dataset/CodeXGlue/code-to-code/test.java-cs.txt.cs'
    ])
    model, test_loader, vocab_tgt, vocab_src = \
        load_CodeBert_model_data(data_path, model_path, beam_size=1)
    model = Wrapper_CodeBert(model, len(vocab_src))
    return model, test_loader, vocab_tgt, vocab_src


def _load_GPT2():
    model, test_loader, vocab_tgt, vocab_src = load_GPT2_model()
    model = Wrapper_GPT2(model, len(vocab_src))
    return model, test_loader, vocab_tgt, vocab_src

#
# def _load_Code2Seq():
#     model, test_loader, vocab_tgt, vocab_src = \
#         load_code2seq_model_data()
#     model = Wrapper_Code2Seq(model, len(vocab_src))
#     return model, test_loader, vocab_tgt, vocab_src


def load_model_data(task_id):
    task_name = TASK_LIST[task_id]
    if task_name == 'deepAPI':
        model, test_loader, vocab_tgt, vocab_src = _load_deepAPI()
    elif task_name == 'CodeBert':
        model, test_loader, vocab_tgt, vocab_src = _load_CodeBert()
    elif task_name == 'GPT2':
        model, test_loader, vocab_tgt, vocab_src = _load_GPT2()
    else:
        raise NotImplementedError
    assert type(vocab_tgt) == dict
    assert type(vocab_tgt) == dict
    return model, test_loader, vocab_tgt, vocab_src
