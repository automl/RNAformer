import numpy as np
import math
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, config):
        super(FeedForward, self).__init__()

        ff_dim = int(config.ff_factor * config.model_dim)

        self.glu = config.use_glu

        if self.glu:
            ff_dim_2 = np.exp2(np.ceil(np.log2(256 * 4 / 3))).astype(int)
            ff_dim_1 = ff_dim_2 * 2
        else:
            ff_dim_1, ff_dim_2 = ff_dim, ff_dim

        self.input_norm = nn.LayerNorm(config.model_dim, eps=config.ln_eps, elementwise_affine=config.learn_ln)

        self.linear_1 = nn.Linear(config.model_dim, ff_dim_1, bias=config.use_bias)
        self.linear_2 = nn.Linear(ff_dim_2, config.model_dim, bias=config.use_bias)
        self.act = nn.SiLU()

        self.initialize(config.zero_init, config.use_bias, config.initializer_range, config.n_layers)

    def initialize(self, zero_init, use_bias, initializer_range, n_layers):

        nn.init.normal_(self.linear_1.weight, mean=0.0, std=initializer_range)

        if use_bias:
            nn.init.constant_(self.linear_1.bias, 0.0)
            nn.init.constant_(self.linear_2.bias, 0.0)

        if zero_init:
            nn.init.constant_(self.linear_2.weight, 0.0)
        else:
            nn.init.normal_(self.linear_2.weight, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers))

    def forward(self, x):

        x = self.input_norm(x)

        if self.glu:
            x = self.linear_1(x)
            x, gate = x.chunk(2, dim=-1)
            x = self.act(gate) * x
        else:
            x = self.act(self.linear_1(x))

        return self.linear_2(x)


class ConvFeedForward(nn.Module):

    def __init__(self, config):
        super(ConvFeedForward, self).__init__()

        ff_dim = int(config.ff_factor * config.model_dim)

        self.zero_init = config.zero_init

        self.input_norm = nn.GroupNorm(1, config.model_dim, affine=config.learn_ln)

        if config.ff_kernel == 1:
            self.conv1 = nn.Conv2d(config.model_dim, ff_dim, kernel_size=1, bias=config.use_bias)
            self.conv2 = nn.Conv2d(ff_dim, config.model_dim, kernel_size=1, bias=config.use_bias)
        else:
            self.conv1 = nn.Conv2d(config.model_dim, ff_dim, bias=config.use_bias, kernel_size=config.ff_kernel,
                                   padding=(config.ff_kernel - 1) // 2)
            self.conv2 = nn.Conv2d(ff_dim, config.model_dim, bias=config.use_bias, kernel_size=config.ff_kernel,
                                   padding=(config.ff_kernel - 1) // 2)

        self.act = nn.SiLU()

        self.initialize(config.zero_init, config.use_bias, config.initializer_range, config.n_layers)

    def initialize(self, zero_init, use_bias, initializer_range, n_layers):

        nn.init.normal_(self.conv1.weight, mean=0.0, std=initializer_range)

        if use_bias:
            nn.init.constant_(self.conv1.bias, 0.0)
            nn.init.constant_(self.conv2.bias, 0.0)

        if zero_init:
            nn.init.constant_(self.conv2.weight, 0.0)
        else:
            nn.init.normal_(self.conv2.weight, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers))

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)
        x = self.input_norm(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        return x
