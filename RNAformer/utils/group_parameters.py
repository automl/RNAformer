import torch.nn as nn

import inspect

from RNAformer.utils import get_class


def group_parameters_for_optimizer(model, optimizer_cfg, bias_regularization=False,
                                   normalization_regularization=False):
    if 'weight_decay' in optimizer_cfg:
        weight_decay = optimizer_cfg.weight_decay
    else:
        signature = inspect.signature(get_class(optimizer_cfg._target_))
        if 'weight_decay' in signature.parameters:
            weight_decay = signature.parameters['weight_decay'].default
            if weight_decay is inspect.Parameter.empty:
                weight_decay = 0.0
        else:
            weight_decay = 0.0

    if weight_decay == 0.0 and not any(hasattr(p, '_optim') for p in model.parameters()):
        return model.parameters()

    skip = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else set()
    skip_keywords = (model.no_weight_decay_keywords() if hasattr(model, 'no_weight_decay_keywords')
                     else set())

    decay = set()
    no_decay = set()
    special = set()
    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (nn.Embedding,)
    if not normalization_regularization:
        blacklist_weight_modules += (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                     nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
                                     nn.GroupNorm, nn.SyncBatchNorm,
                                     nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                     nn.LayerNorm, nn.LocalResponseNorm)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if not p.requires_grad or fpn not in param_dict:
                continue
            if hasattr(p, '_optim'):
                special.add(fpn)
            elif fpn in skip or any(skip_keyword in fpn for skip_keyword in skip_keywords):
                no_decay.add(fpn)
            elif getattr(p, '_no_weight_decay', False):
                no_decay.add(fpn)
            elif not bias_regularization and pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    decay |= (param_dict.keys() - no_decay - special)
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(
        param_dict.keys() - special - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)}  were not separated into either decay/no_decay set!"

    if weight_decay == 0.0 or not no_decay:
        param_groups = [{"params": [param_dict[pn] for pn in sorted(list(no_decay | decay))],
                         "weight_decay": weight_decay}]
    else:
        param_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
    hps = [dict(s) for s in set(frozenset(param_dict[pn]._optim.items()) for pn in special)]
    for hp in hps:
        params = [param_dict[pn] for pn in sorted(list(special)) if param_dict[pn]._optim == hp]
        param_groups.append({"params": params, **hp})

    return param_groups
