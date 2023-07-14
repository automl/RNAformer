import os
import tqdm
import argparse
import pathlib
import torch
import urllib.request
import logging

import torch.cuda
import numpy as np
import pandas as pd

from RNAformer.model.RNAformer import RiboFormer
from RNAformer.pl_module.datamodule_rna import IGNORE_INDEX, PAD_INDEX
from RNAformer.utils.data.rna import CollatorRNA
from RNAformer.utils.configuration import Config
from eval_predictions import evaluate, print_dict_tables

logger = logging.getLogger(__name__)


class EvalRNAformer():

    def __init__(self, model_dir, precision=16, flash_attn=False):

        model_dir = pathlib.Path(model_dir)

        config = Config(config_file=model_dir / 'config.yml')
        state_dict = torch.load(model_dir / 'state_dict.pth')

        if precision == 32 or flash_attn == False:
            config.trainer.precision = 32
            config.RNAformer.precision = 32
            config.RNAformer.flash_attn = False
        elif precision == 16 or precision == 'fp16':
            config.trainer.precision = 16
            config.RNAformer.precision = 16
            config.RNAformer.flash_attn = True
        elif precision == 'bf16':
            config.trainer.precision = 'bf16'
            config.RNAformer.precision = 'bf16'
            config.RNAformer.flash_attn = True

        model_config = config.RNAformer
        model_config.seq_vocab_size = 5
        model_config.max_len = state_dict["seq2mat_embed.src_embed_1.embed_pair_pos.weight"].shape[1]

        model = RiboFormer(model_config)
        model.load_state_dict(state_dict, strict=True)

        model = model.cuda()
        if precision == 16 or precision == 'fp16' or precision == 'bf16':
            model = model.half()
        self.model = model.eval()

        self.ignore_index = IGNORE_INDEX
        self.pad_index = PAD_INDEX

        self.collator = CollatorRNA(self.pad_index, self.ignore_index)

    def __call__(self, sequence: str, mean_triual=True):
        length = len(sequence)

        seq_vocab = ['A', 'C', 'G', 'U', 'N']
        seq_stoi = dict(zip(seq_vocab, range(len(seq_vocab))))
        int_sequence = list(map(seq_stoi.get, sequence))
        input_sample = torch.LongTensor(int_sequence)

        input_sample = {'src_seq': input_sample, 'length': torch.LongTensor([len(input_sample)])[0]}
        batch = self.collator([input_sample])
        with torch.no_grad():
            logits, mask = self.model(batch['src_seq'].cuda(), batch['length'].cuda(), infer_mean=True)
        sample_logits = logits[0, :length, :length, -1].detach()
        # triangle mask
        if mean_triual:
            low_tr = torch.tril(sample_logits, diagonal=-1)
            upp_tr = torch.triu(sample_logits, diagonal=1)
            mean_logits = (low_tr.t() + upp_tr) / 2
            sample_logits = mean_logits + mean_logits.t()

        pred_mat = torch.sigmoid(sample_logits) > 0.5

        return pred_mat.cpu().numpy()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train RNAformer')
    parser.add_argument('-n', '--model_name', type=str, default="ts0_conform_dim256_32bit")
    parser.add_argument('-m', '--model_dir', type=str, )
    parser.add_argument('-f', '--flash_attn', type=bool, default=False )
    parser.add_argument('-p', '--precision', type=int, default=32 )
    parser.add_argument('-s', '--save_predictions', type=bool, default=False )

    args, unknown_args = parser.parse_known_args()

    if args.model_dir is None:
        model_dir = f"checkpoints/{args.model_name}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            print("Downloading model checkpoints")
            urllib.request.urlretrieve(
                f"https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/{args.model_name}/config.yml",
                f"checkpoints/{args.model_name}/config.yml")
            urllib.request.urlretrieve(
                f"https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/{args.model_name}/state_dict.pth",
                f"checkpoints/{args.model_name}/state_dict.pth")
    else:
        model_dir = args.model_dir


    eval_model = EvalRNAformer(model_dir, precision=args.precision, flash_attn=args.flash_attn)

    def count_parameters(parameters):
        return sum(p.numel() for p in parameters)
    print(f"Model size: {count_parameters(eval_model.model.parameters())}")

    file = "data/all_test_data.npy"
    df = pd.DataFrame(np.load(file, allow_pickle=True).tolist())

    # potential test sets: 'bprna_ts0', 'pdb_ts1', 'pdb_ts2', 'pdb_ts3', 'pdb_ts_hard', 'synthetic_test'
    data_sets = ['bprna_ts0']

    for dataset in data_sets:
        print("Evaluating on", dataset)
        processed_samples = []
        for id, sample in tqdm.tqdm(enumerate(df[df['dataset'] == dataset].to_dict('records'))):
            sequence = sample['sequence']
            pred_mat = eval_model(sequence, mean_triual=True)
            sample['struct_mat'] = pred_mat
            processed_samples.append(sample)
        result = evaluate(processed_samples)
        print_dict_tables(result)

    if args.save_predictions:
        os.makedirs("predictions", exist_ok=True)
        np.save(f"predictions/{args.model_name}.npy", processed_samples)
