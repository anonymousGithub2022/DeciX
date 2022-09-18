from transformers import AutoTokenizer, AutoModelForCausalLM
import csv


def load_GPT2_model():
    tokenizer = AutoTokenizer.from_pretrained("SIC98/GPT2-python-code-generator")
    model = AutoModelForCausalLM.from_pretrained("SIC98/GPT2-python-code-generator")

    with open("./data/PyGPT2/test.csv", 'r') as csvfile:
        data = csvfile.readlines()
    data = ''.join(data[1:]).split('<EOS>')
    data = [d.replace('<BOS>', '') for d in data]

    data = [d.split('\n') for d in data]
    new_data = []
    for d in data:
        new_data.extend(d)
    new_data = [d for d in new_data if 20 < len(d) < 200]

    data_loader = [
        tokenizer.encode(s, return_tensors='pt')
        for s in new_data
    ]
    model.tokenizer = tokenizer
    model.max_length = model.config.max_length

    # new_vocab = {}
    # for k in tokenizer.vocab:
    #     new_vocab[tokenizer.vocab[k]] = k
    return model, data_loader, tokenizer.vocab, tokenizer.vocab
