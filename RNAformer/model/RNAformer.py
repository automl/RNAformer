import torch
import torch.nn as nn

from RNAformer.module.embedding import EmbedSequence2Matrix
from RNAformer.model.RNAformer_stack import RNAformerStack


class RiboFormer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model_dim = config.model_dim

        if hasattr(config, "cycling") and config.cycling:
            self.initialize_cycling(config.cycling)
        else:
            self.cycling = False

        self.seq2mat_embed = EmbedSequence2Matrix(config)
        self.RNAformer = RNAformerStack(config)

        if not hasattr(config, "pdb_flag") or config.pdb_flag:
            self.pdf_embedding = nn.Linear(1, config.model_dim, bias=True)
            self.use_pdb = True
        else:
            self.use_pdb = False

        if not hasattr(config, "binary_output") or config.binary_output:
            self.output_mat = nn.Linear(config.model_dim, 1, bias=True)
        else:
            self.output_mat = nn.Linear(config.model_dim, 2, bias=False)

        self.initialize(initializer_range=config.initializer_range)

    def initialize(self, initializer_range):

        nn.init.normal_(self.output_mat.weight, mean=0.0, std=initializer_range)

    def initialize_cycling(self, cycle_steps):
        import random
        self.cycling = True
        self.cycle_steps = cycle_steps
        self.recycle_pair_norm = nn.LayerNorm(self.model_dim, elementwise_affine=True)
        self.trng = torch.Generator()
        self.trng.manual_seed(random.randint(1, 10000))

    def make_pair_mask(self, src, src_len):
        encode_mask = torch.arange(src.shape[1], device=src.device).expand(src.shape[:2]) < src_len.unsqueeze(1)

        pair_mask = encode_mask[:, None, :] * encode_mask[:, :, None]

        assert isinstance(pair_mask, torch.BoolTensor) or isinstance(pair_mask, torch.cuda.BoolTensor)
        return torch.bitwise_not(pair_mask)

    @torch.no_grad()
    def cycle_riboformer(self, pair_act, pair_mask):
        latent = self.RNAformer(pair_act=pair_act, pair_mask=pair_mask, cycle_infer=True)
        return latent.detach()

    def forward(self, src_seq, src_len, pdb_sample, max_cycle=0):

        pair_mask = self.make_pair_mask(src_seq, src_len)

        pair_latent = self.seq2mat_embed(src_seq)

        if self.use_pdb:
            pair_latent = pair_latent + self.pdf_embedding(pdb_sample)[:, None, None, :]

        if self.cycling:
            if self.training:
                n_cycles = torch.randint(0, max_cycle + 1, [1])
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    n_cycles = n_cycles.to(torch.int64).cuda()
                    tensor_list = [torch.zeros(1, dtype=torch.int64).cuda() for _ in
                                   range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(tensor_list, n_cycles)
                    n_cycles = tensor_list[0]
                n_cycles = n_cycles.item()
            else:
                n_cycles = self.cycle_steps

            cyc_latent = torch.zeros_like(pair_latent)
            for n in range(n_cycles - 1):
                res_latent = pair_latent.detach() + self.recycle_pair_norm(cyc_latent.detach()).detach()
                cyc_latent = self.cycle_riboformer(res_latent, pair_mask.detach())

            pair_latent = pair_latent + self.recycle_pair_norm(cyc_latent.detach())

        latent = self.RNAformer(pair_act=pair_latent, pair_mask=pair_mask, cycle_infer=False)

        logits = self.output_mat(latent)

        return logits, pair_mask
