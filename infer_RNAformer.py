import os
import argparse
import torch
import urllib.request
import logging
from collections import defaultdict
import torch.cuda
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Infer RNAformer')
    parser.add_argument('-s', '--sequence', type=str, default=None)
    parser.add_argument('--state_dict', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('-c', '--cycling', type=int, default=0)

    args, unknown_args = parser.parse_known_args()

    config = Config(config_file=args.config)

    if args.cycling:
        config.RNAformer.cycling = args.cycling

    model = RiboFormer(config.RNAformer)

    if hasattr(config, "lora") and config.lora:
        model = insert_lora_layer(model, config)

    state_dict = torch.load(args.state_dict)

    model.load_state_dict(state_dict, strict=True)

    if args.cycling and args.cycling > 0:
        model.cycle_steps = args.cycling

    model_name = args.state_dict.split(".pth")[0]

    model = model.cuda()

    # check GPU can do bf16
    if torch.cuda.is_bf16_supported():
        model = model.bfloat16()
    else:
        model = model.half()

    model.eval()

    with torch.no_grad():

        seq_vocab = ['A', 'C', 'G', 'U', 'N']
        seq_stoi = dict(zip(seq_vocab, range(len(seq_vocab))))

        sequence = args.sequence
        pdb_sample = 1

        length = len(sequence)
        src_seq = sequence2index_vector(sequence, seq_stoi)

        sample = {}
        sample['src_seq'] = src_seq.clone()
        sample['length'] = torch.LongTensor([length])[0]
        sample['pdb_sample'] = torch.LongTensor([pdb_sample])[0]

        sequence = sample['src_seq'].unsqueeze(0).cuda()
        src_len = torch.LongTensor([sequence.shape[-1]]).cuda()
        if torch.cuda.is_bf16_supported():
            pdb_sample = torch.FloatTensor([[1]]).bfloat16().cuda()
        else:
            pdb_sample = torch.FloatTensor([[1]]).half().cuda()

        logits, pair_mask = model(sequence, src_len, pdb_sample)

        pred_mat = torch.sigmoid(logits[0, :, :, -1]) > 0.5

    pos_id = torch.where(pred_mat == True)
    pos1_id = pos_id[0].cpu().tolist()
    pos2_id = pos_id[1].cpu().tolist()
    print("Pairing index 1:", pos1_id)
    print("Pairing index 2:", pos2_id)
