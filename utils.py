import os
from src import *


TASK_LIST = [
    'deepAPI',
    'CodeBert',
    'GPT2'
]

EXPLAIN_LIST = [
    DepCausalExp,         # full Markov dependency + casuality
    NoDepCausalExp,       # Np dependency decomposition + casuality
    DepLinearExp,         # full Markov dependency  + linear

    RandomExp,            # random explanation

    LimeExp,  # lime explanation
    LemnaExp,  # lemna explanation
]

'''
    deepAPI: 
        x: 1, X, X, 2.      len = 2  
        y: X, X, X, 2.      len = 4
    CodeBert:
        x: 0, X, X, 2.      len = 4
        y: 0, X, X, 2.      len = 4
    GPT2
        x: X, X, X, X.      len = 4
        y: X, X, X, X.      len = 4
'''


def _load_deepAPI():
    data_dir = './data/deepAPI/'
    model_path = './model_weight/deepAPI/model_epo120000.pkl'
    model, test_loader, vocab_api, vocab_desc =\
        load_deepAPI_model_data(data_dir, model_path)
    model = Wrapper_DeepAPI(model, len(vocab_desc))
    return model, test_loader, vocab_api, vocab_desc


def _load_CodeBert():
    model_path = './model_weight/pytorch_model.bin'
    data_path = ''.join([
        './data/code-to-code/test.java-cs.txt.java'
        ','
        './data/code-to-code/test.java-cs.txt.cs'
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


