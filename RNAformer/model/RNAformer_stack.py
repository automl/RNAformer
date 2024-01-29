import torch.nn as nn

from RNAformer.model.RNAformer_block import RNAformerBlock


class RNAformerStack(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.output_ln = nn.LayerNorm(config.model_dim, eps=config.ln_eps, elementwise_affine=config.learn_ln)

        module_list = []
        for idx in range(config.n_layers):
            layer = RNAformerBlock(config=config)
            module_list.append(layer)
        self.layers = nn.ModuleList(module_list)

    def forward(self, pair_act, pair_mask, cycle_infer=False):

        for idx, layer in enumerate(self.layers):
            pair_act = layer(pair_act, pair_mask, cycle_infer=cycle_infer)

        pair_act = self.output_ln(pair_act)

        return pair_act
