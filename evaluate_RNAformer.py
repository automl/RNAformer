import os
from tqdm import tqdm
import argparse
import torch
import urllib.request
import logging
from collections import defaultdict
import torch.cuda
import pandas as pd
import loralib as lora

from RNAformer.model.RNAformer import RiboFormer
from RNAformer.utils.configuration import Config

logger = logging.getLogger(__name__)


def insert_lora_layer(model, ft_config):
    lora_config = {
        "r": ft_config.r,
        "lora_alpha": ft_config.lora_alpha,
        "lora_dropout": ft_config.lora_dropout,
    }

    with torch.no_grad():
        for name, module in model.named_modules():
            if any(replace_key in name for replace_key in ft_config.replace_layer):
                parent = model.get_submodule(".".join(name.split(".")[:-1]))
                target_name = name.split(".")[-1]
                target = model.get_submodule(name)
                if isinstance(target, torch.nn.Linear) and "qkv" in name:
                    new_module = lora.MergedLinear(target.in_features, target.out_features,
                                                   bias=target.bias is not None,
                                                   enable_lora=[True, True, True], **lora_config)
                    new_module.weight.copy_(target.weight)
                    if target.bias is not None:
                        new_module.bias.copy_(target.bias)
                elif isinstance(target, torch.nn.Linear):
                    new_module = lora.Linear(target.in_features, target.out_features,
                                             bias=target.bias is not None, **lora_config)
                    new_module.weight.copy_(target.weight)
                    if target.bias is not None:
                        new_module.bias.copy_(target.bias)

                elif isinstance(target, torch.nn.Conv2d):
                    kernel_size = target.kernel_size[0]
                    new_module = lora.Conv2d(target.in_channels, target.out_channels, kernel_size,
                                             padding=(kernel_size - 1) // 2, bias=target.bias is not None,
                                             **lora_config)

                    new_module.conv.weight.copy_(target.weight)
                    if target.bias is not None:
                        new_module.conv.bias.copy_(target.bias)
                setattr(parent, target_name, new_module)

    return model


def sequence2index_vector(sequence, mapping):
    int_sequence = list(map(mapping.get, sequence))
    return torch.LongTensor(int_sequence)


def prepare_RNA_sample(input_sample):
    seq_vocab = ['A', 'C', 'G', 'U', 'N']
    seq_stoi = dict(zip(seq_vocab, range(len(seq_vocab))))

    sequence = input_sample["sequence"]
    pos1id = input_sample["pos1id"]
    pos2id = input_sample["pos2id"]
    pdb_sample = int(input_sample['is_pdb'])

    length = len(sequence)

    src_seq = sequence2index_vector(sequence, seq_stoi)

    torch_sample = {}
    torch_sample['src_seq'] = src_seq.clone()
    torch_sample['length'] = torch.LongTensor([length])[0]
    torch_sample['pos1id'] = torch.LongTensor(pos1id)
    torch_sample['pos2id'] = torch.LongTensor(pos2id)
    torch_sample['pdb_sample'] = torch.LongTensor([pdb_sample])[0]

    trg_mat = torch.LongTensor(length, length).fill_(0)
    trg_mat[pos1id, pos2id] = 1
    trg_mat[pos2id, pos1id] = 1
    torch_sample['trg_mat'] = trg_mat
    torch_sample['trg_seq'] = src_seq.clone()

    return torch_sample


def evaluate_RNAformer(model, test_sets, eval_synthetic=False, eval_bprna=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        model = model.cuda()

        # check GPU can do bf16
        if torch.cuda.is_bf16_supported():
            model = model.bfloat16()
        else:
            model = model.half()

    model.eval()

    test_results = []
    all_samples = {}

    if eval_synthetic:
        eval_sets = ["synthetic_test"]
    elif eval_bprna:
        eval_sets = ["bprna_ts0"]
    else:
        eval_sets = ["pdb_ts1", "pdb_ts2", "pdb_ts3", "pdb_ts_hard"]

    with torch.no_grad():
        for test_set, df in test_sets.items():

            if test_set not in eval_sets:
                continue

            print("Evaluating on", test_set)
            test_df = df[df['set'].str.contains(test_set)]
            test_df = test_df.reset_index()
            metrics = defaultdict(list)

            inference_samples = []

            for id, sample_raw in tqdm(test_df.iterrows(), total=len(test_df)):

                sample = prepare_RNA_sample(sample_raw)

                sequence = sample['src_seq'].unsqueeze(0).to(device)
                src_len = torch.LongTensor([sequence.shape[-1]]).to(device)

                if torch.cuda.is_available():
                    if torch.cuda.is_bf16_supported():
                        pdb_sample = torch.FloatTensor([[1]]).bfloat16().cuda()
                    else:
                        pdb_sample = torch.FloatTensor([[1]]).half().cuda()
                else:
                     pdb_sample = torch.FloatTensor([[1]]).to(device)
                logits, pair_mask = model(sequence, src_len, pdb_sample)

                pred_mat = torch.sigmoid(logits[0, :, :, -1]) > 0.5
                true_mat = sample['trg_mat'].float().to(device)

                save_sample = {}
                for nkey in ['sequence', 'pk', 'has_multiplet', 'has_pk', 'set', 'has_nc']:
                    save_sample[nkey] = sample_raw[nkey]
                save_sample['true_mat'] = sample['trg_mat']
                save_sample['pred_mat'] = pred_mat.detach().cpu()
                save_sample['logits_mat'] = torch.sigmoid(logits[0, :, :, 0]).detach().cpu()
                inference_samples.append(save_sample)

                solved = torch.equal(true_mat, pred_mat).__int__()

                metrics['solved'].append(torch.tensor([solved], dtype=true_mat.dtype, device=true_mat.device))

                tp = torch.logical_and(pred_mat, true_mat).sum()
                tn = torch.logical_and(torch.logical_not(pred_mat), torch.logical_not(true_mat)).sum()
                fp = torch.logical_and(pred_mat, torch.logical_not(true_mat)).sum()
                fn = torch.logical_and(torch.logical_not(pred_mat), true_mat).sum()
                assert pred_mat.size().numel() == tp + tn + fp + fn
                accuracy = tp / pred_mat.size().numel()
                precision = tp / (1e-4 + tp + fp)
                recall = tp / (1e-4 + tp + fn)
                f1_score = 2 * tp / (1e-4 + (2 * tp + fp + fn))
                mcc = (tp * tn - fp * fn) / (1e-4 + torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
                metrics['accuracy'].append(accuracy)
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1_score'].append(f1_score)
                metrics['mcc'].append(mcc)

            num_samples = len(metrics['mcc'])
            for key, value_list in metrics.items():
                metrics[key] = torch.stack(value_list).mean().item()
            metrics['num_samples'] = num_samples

            metrics['test_set'] = test_set
            test_results.append(metrics)
            all_samples[test_set] = inference_samples

    test_results = pd.DataFrame(test_results)

    return test_results, all_samples


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate RNAformer')
    parser.add_argument('--state_dict', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('-s', '--save_predictions', action='store_true')
    parser.add_argument('-y', '--eval_synthetic', action='store_true')
    parser.add_argument('-b', '--eval_eval_bprna', action='store_true')
    parser.add_argument('-c', '--cycling', type=int, default=0)

    args, unknown_args = parser.parse_known_args()

    config = Config(config_file=args.config)

    if args.cycling:
        config.RNAformer.cycling = args.cycling

    model = RiboFormer(config.RNAformer)

    if hasattr(config, "lora") and config.lora:
        model = insert_lora_layer(model, config)

    if torch.cuda.is_available():
            state_dict = torch.load(args.state_dict)
    else:
        state_dict = torch.load(args.state_dict, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict, strict=True)

    if args.cycling and args.cycling > 0:
        model.cycle_steps = args.cycling

    model_name = args.state_dict.split(".pth")[0]


    def count_parameters(parameters):
        return sum(p.numel() for p in parameters)


    print(f"Model size: {count_parameters(model.parameters()):,d}")

    if not os.path.exists("datasets/test_sets.plk"):
        print("Downloading test sets")
        os.makedirs("datasets", exist_ok=True)
        urllib.request.urlretrieve(
            "https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/test_sets.plk",
            "datasets/test_sets.plk")

    test_sets = pd.read_pickle("datasets/test_sets.plk")

    test_results, all_samples = evaluate_RNAformer(model, test_sets, eval_synthetic=args.eval_synthetic,
                                                   eval_bprna=args.eval_eval_bprna)

    if args.save_predictions:
        os.makedirs("predictions", exist_ok=True)
        torch.save(all_samples, os.path.join("predictions", f"{model_name}_test_samples.pt"))
        pd.to_pickle(test_results, os.path.join("predictions", f"{model_name}_test_results.plk"))

    print(test_results.to_markdown())
