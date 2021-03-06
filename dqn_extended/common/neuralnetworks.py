import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):

        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256.0
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions, noisy_nets_sigma):

        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(
            NoisyLinear(conv_out_size, 512, noisy_nets_sigma),
            nn.ReLU(),
            NoisyLinear(512, n_actions, noisy_nets_sigma),
        )

        self.fc_val = nn.Sequential(
            NoisyLinear(conv_out_size, 512, noisy_nets_sigma),
            nn.ReLU(),
            NoisyLinear(512, 1, noisy_nets_sigma),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256.0
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init)
        )
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        return F.linear(
            input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias
        )


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """

    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(
            in_features, out_features, bias=bias
        )
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init)
        )
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def forward(self, input):
        self.epsison_input.normal_()
        self.epsilon_output.normal_()

        def func(x):
            return torch.sign(x) * torch.sqrt(torch.abs(x))

        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions, n_atoms, supports):
        super(RainbowDQN, self).__init__()
        self.n_atoms = n_atoms

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_val = nn.Sequential(
            NoisyLinear(conv_out_size, 512), nn.ReLU(), NoisyLinear(512, n_atoms)
        )

        self.fc_adv = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions * n_atoms),
        )

        self.register_buffer("supports", supports)
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1, self.n_atoms)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, self.n_atoms)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, self.n_atoms)).view(t.size())


class RainbowQRDQN(nn.Module):
    def __init__(self, input_shape, n_actions, noisy_nets_sigma, n_quantiles):
        super(RainbowQRDQN, self).__init__()
        self.n_quantiles = n_quantiles

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            NoisyLinear(conv_out_size, 512, noisy_nets_sigma),
            nn.ReLU(),
            NoisyLinear(512, n_actions * n_quantiles, noisy_nets_sigma),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 255
        conv_out = self.conv(fx).view(batch_size, -1)
        return self.fc(conv_out).view(batch_size, -1, self.n_quantiles)

    def qvals(self, x):
        return self.qvals_from_quant(self(x))

    def qvals_from_quant(self, quantiles):
        return quantiles.mean(dim=2)


class QRDQN(nn.Module):
    def __init__(self, input_shape, n_actions, n_quantiles):
        super(QRDQN, self).__init__()
        self.n_quantiles = n_quantiles

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * n_quantiles),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 255
        conv_out = self.conv(fx).view(batch_size, -1)
        return self.fc(conv_out).view(batch_size, -1, self.n_quantiles)

    def qvals(self, x):
        return self.qvals_from_quant(self(x))

    def qvals_from_quant(self, quantiles):
        return quantiles.mean(dim=2)
