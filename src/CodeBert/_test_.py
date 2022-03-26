import torch

from .run import *


def my_convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length
        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


model_type = 'roberta'
model_name = 'roberta-base'
tokenizer_name = 'roberta-base'
beam_size = 10
max_source_length, max_target_length = 512, 512
device = torch.device(4)

config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
config = config_class.from_pretrained(model_name)
tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=True)

encoder = model_class.from_pretrained(model_name, config=config)
decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                beam_size=10, max_length=max_target_length,
                sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

load_model_path = '/home/sxc180080/data/Project/CodeGenExp/src/CodeBert/roberta/checkpoint-best-bleu/pytorch_model.bin'
model.load_state_dict(torch.load(load_model_path))
files = []
test_filename = ''.join([
    '/home/sxc180080/data/Project/Dataset/CodeXGlue/code-to-code/test.java-cs.txt.java'
    ','
    '/home/sxc180080/data/Project/Dataset/CodeXGlue/code-to-code/test.java-cs.txt.cs'
])

files.append(test_filename)
for idx, file in enumerate(files):
    eval_examples = read_examples(file)
    eval_features = my_convert_examples_to_features(eval_examples, tokenizer, max_source_length, max_target_length, stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids, all_source_mask)

    # Calculate bleu
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    p = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask = batch
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)
    # model.train()
    # predictions = []
    # accs = []
    # with open(os.path.join(output_dir, "test_{}.output".format(str(idx))), 'w') as f, open(
    #         os.path.join(output_dir, "test_{}.gold".format(str(idx))), 'w') as f1:
    #     for ref, gold in zip(p, eval_examples):
    #         predictions.append(str(gold.idx) + '\t' + ref)
    #         f.write(ref + '\n')
    #         f1.write(gold.target + '\n')
    #         accs.append(ref == gold.target)
    # dev_bleu = round(_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(file),
    #                        os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(file)), 2)
