import torch.nn as nn

from RNAformer.module.feed_forward import FeedForward, ConvFeedForward
from RNAformer.module.axial_attention import AxialAttention, TriangleAttention


class RNAformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        if config.rotary_emb:
            self.attn_pair_row = AxialAttention(config, 'per_row')
            self.attn_pair_col = AxialAttention(config, 'per_column')
        else:
            self.attn_pair_row = TriangleAttention(config.model_dim, config.num_head, 'per_row', config.softmax_scale,
                                                   config.precision, config.zero_init, config.use_bias,
                                                   config.flash_attn,
                                                   config.initializer_range, config.n_layers)
            self.attn_pair_col = TriangleAttention(config.model_dim, config.num_head, 'per_column',
                                                   config.softmax_scale,
                                                   config.precision, config.zero_init, config.use_bias,
                                                   config.flash_attn,
                                                   config.initializer_range, config.n_layers)

        self.pair_dropout_row = nn.Dropout(p=config.resi_dropout / 2)
        self.pair_dropout_col = nn.Dropout(p=config.resi_dropout / 2)

        if config.ff_kernel:
            self.pair_transition = ConvFeedForward(config)
        else:
            self.pair_transition = FeedForward(config)

        self.res_dropout = nn.Dropout(p=config.resi_dropout)

    def forward(self, pair_act, pair_mask, cycle_infer=False):

        pair_act = pair_act + self.pair_dropout_row(self.attn_pair_row(pair_act, pair_mask, cycle_infer))
        pair_act = pair_act + self.pair_dropout_col(self.attn_pair_col(pair_act, pair_mask, cycle_infer))
        pair_act = pair_act + self.res_dropout(self.pair_transition(pair_act))

        return pair_act
