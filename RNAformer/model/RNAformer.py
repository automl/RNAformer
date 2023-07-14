import torch
import torch.nn as nn

from RNAformer.module.embedding import EmbedSequence2Matrix
from RNAformer.model.RNAformer_stack import RNAformerStack


class RiboFormer(nn.Module):

    def __init__(self, config):
        super().__init__()

        if hasattr(config, "cycling") and config.cycling:
            import random
            self.cycling = True
            self.cycle_steps = config.cycling
            self.recycle_pair_norm = nn.LayerNorm(config.model_dim)
            self.trng = torch.Generator()
            self.trng.manual_seed(random.randint(1, 10000))
        else:
            self.cycling = False

        self.seq2mat_embed = EmbedSequence2Matrix(config)
        self.RNAformer = RNAformerStack(config)

        self.output_mat = nn.Linear(config.model_dim, 2, bias=False)

        self.initialize(initializer_range=config.initializer_range)

    def initialize(self, initializer_range):

        nn.init.normal_(self.output_mat.weight, mean=0.0, std=initializer_range)

    def make_pair_mask(self, src, src_len):
        encode_mask = torch.arange(src.shape[1], device=src.device).expand(src.shape[:2]) < src_len.unsqueeze(1)

        pair_mask = encode_mask[:, None, :] * encode_mask[:, :, None]

        assert isinstance(pair_mask, torch.BoolTensor) or isinstance(pair_mask, torch.cuda.BoolTensor)
        return torch.bitwise_not(pair_mask)

    @torch.no_grad()
    def cycle_riboformer(self, latent2, pair_mask):
        latent = self.RNAformer(pair_act=latent2, pair_mask=pair_mask)
        return latent.detach()

    def forward(self, src_seq=None, src_len=None, infer_mean=False):

        pair_mask = self.make_pair_mask(src_seq, src_len)

        pair_latent = self.seq2mat_embed(src_seq)

        pair_latent.masked_fill_(pair_mask[:, :, :, None], 0.0)

        if self.cycling:
            if self.training:
                n_cycles = torch.randint(2, self.cycle_steps + 1, [1])  # , generator=self.trng).item()
            else:
                n_cycles = self.cycle_steps

            # print("N", n_cycles)
            latent = torch.zeros_like(pair_latent, requires_grad=True)
            with torch.no_grad():
                for n in range(n_cycles - 1):
                    res_latent = pair_latent.detach() + self.recycle_pair_norm(latent.detach())
                    latent = self.RNAformer(pair_act=res_latent, pair_mask=pair_mask, cycle_infer=True)

            res_latent = pair_latent + self.recycle_pair_norm(latent)
            latent = self.RNAformer(pair_act=res_latent, pair_mask=pair_mask, cycle_infer=False)
        else:
            latent = self.RNAformer(pair_act=pair_latent, pair_mask=pair_mask, cycle_infer=False)

        logits = self.output_mat(latent)

        return logits, pair_mask
