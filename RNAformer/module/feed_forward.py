import numpy as np
import math
import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, model_dim, ff_dim, use_bias, glu, initializer_range, zero_init, n_layers):
        super(FeedForward, self).__init__()

        self.glu = glu

        if self.glu:
            ff_dim_2 = np.exp2(np.ceil(np.log2(256 * 4 / 3))).astype(int)
            ff_dim_1 = ff_dim_2 * 2
        else:
            ff_dim_1, ff_dim_2 = ff_dim, ff_dim

        self.input_norm = nn.LayerNorm(model_dim, eps=1e-6)

        self.linear_1 = nn.Linear(model_dim, ff_dim_1, bias=use_bias)
        self.linear_2 = nn.Linear(ff_dim_2, model_dim, bias=use_bias)
        self.act = nn.SiLU()

        self.initialize(zero_init, use_bias, initializer_range, n_layers)

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

    def __init__(self, model_dim, ff_dim, use_bias, initializer_range, n_layers, kernel, zero_init=True):
        super(ConvFeedForward, self).__init__()

        self.zero_init = zero_init

        self.input_norm = nn.GroupNorm(1, model_dim)

        if kernel == 1:
            self.conv1 = nn.Conv2d(model_dim, ff_dim, kernel_size=1, bias=use_bias)
            self.conv2 = nn.Conv2d(ff_dim, model_dim, kernel_size=1, bias=use_bias)
        else:
            self.conv1 = nn.Conv2d(model_dim, ff_dim, bias=use_bias, kernel_size=kernel, padding=(kernel - 1) // 2)
            self.conv2 = nn.Conv2d(ff_dim, model_dim, bias=use_bias, kernel_size=kernel, padding=(kernel - 1) // 2)

        self.act = nn.SiLU()

        self.initialize(zero_init, use_bias, initializer_range, n_layers)

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
