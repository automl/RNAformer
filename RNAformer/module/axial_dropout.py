import torch
import torch.nn as nn


class AxialDropout(nn.Module):

    def __init__(self, p, orientation):

        super().__init__()

        if p < 0.0 or p > 1.0:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

        assert orientation in ['per_row', 'per_column']
        self.orientation = orientation

        self.keep_rate = 1.0 - self.p
        self.binomial = torch.distributions.binomial.Binomial(probs=self.keep_rate)

    def forward(self, tensor):

        if self.orientation == 'per_row':
            broadcast_dim = 1
        else:
            broadcast_dim = 2

        if self.training and self.p > 0.0:
            shape = list(tensor.shape)
            shape[broadcast_dim] = 1

            return self.binomial.sample(torch.Size(shape)).to(tensor.device).to(tensor.dtype) * tensor / self.keep_rate
        else:
            return tensor
