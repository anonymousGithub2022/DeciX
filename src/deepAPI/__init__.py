import torch
from tqdm import tqdm

from .myhelper import indexes2sent
from .rnn_seq2seq import *
from .metrics import Metrics
from .data_loader import APIDataset, load_dict, load_vecs
from .configs import config_RNNEncDec


def load_deepAPI_model_data(data_path, model_path, batch_size=1):
    conf = config_RNNEncDec()
    test_set = APIDataset(data_path + 'test.desc.h5', data_path + 'test.apiseq.h5', conf['max_sent_len'])
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=1)
    vocab_api = load_dict(data_path + 'vocab.apiseq.json')
    vocab_desc = load_dict(data_path + 'vocab.desc.json')

    model = RNNEncDec(conf)
    model.load_state_dict(torch.load(model_path))
    return model, test_loader, vocab_api, vocab_desc


def evaluate_deepAPI(model, metrics, test_loader, vocab_desc, vocab_api, repeat, decode_mode, f_eval):
    device = next(model.parameters()).device

    recall_bleus, prec_bleus = [], []
    local_t = 0
    for descs, desc_lens, apiseqs, api_lens in tqdm(test_loader):
        if local_t > 1000:
            break

        desc_str = indexes2sent(descs[0].numpy(), vocab_desc)

        descs, desc_lens = [tensor.to(device) for tensor in [descs, desc_lens]]
        with torch.no_grad():
            sample_words, sample_lens = model.sample(descs, desc_lens, repeat, decode_mode)
        # nparray: [repeat x seq_len]
        pred_sents, _ = indexes2sent(sample_words, vocab_api)
        pred_tokens = [sent.split(' ') for sent in pred_sents]
        ref_str, _ = indexes2sent(apiseqs[0].numpy(), vocab_api)
        ref_tokens = ref_str.split(' ')

        max_bleu, avg_bleu = metrics.sim_bleu(pred_tokens, ref_tokens)
        recall_bleus.append(max_bleu)
        prec_bleus.append(avg_bleu)

        local_t += 1
        f_eval.write("Batch %d \n" % (local_t))  # print the context
        f_eval.write(f"Query: {desc_str} \n")
        f_eval.write("Target >> %s\n" % (ref_str.replace(" ' ", "'")))  # print the true outputs
        for r_id, pred_sent in enumerate(pred_sents):
            f_eval.write("Sample %d >> %s\n" % (r_id, pred_sent.replace(" ' ", "'")))
        f_eval.write("\n")

    recall_bleu = float(np.mean(recall_bleus))
    prec_bleu = float(np.mean(prec_bleus))
    f1 = 2 * (prec_bleu * recall_bleu) / (prec_bleu + recall_bleu + 10e-12)

    report = "Avg recall BLEU %f, avg precision BLEU %f, F1 %f" % (recall_bleu, prec_bleu, f1)
    print(report)
    f_eval.write(report + "\n")
    print("Done testing")

    return recall_bleu, prec_bleu


def test_deepAPI(model, metrics, test_loader, vocab_desc, vocab_api, repeat, decode_mode, f_eval):
    device = next(model.parameters()).device
    for descs, desc_lens, apiseqs, api_lens in tqdm(test_loader):
        descs, desc_lens = [tensor.to(device) for tensor in [descs, desc_lens]]

        sample_words, sample_lens, _ = model.predict_likehood(descs, desc_lens, repeat, decode_mode)