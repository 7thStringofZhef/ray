import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, same_padding, \
    SlimConv2d, SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, try_import_torch

import torch
import torch.nn as nn
#_, nn = try_import_torch()

# Model config fields
OCNET_DEFAULT_CONFIG = {
    'oc_num_options': 4,
    'oc_option_epsilon': 0.1
}

# Atari filters as in https://arxiv.org/pdf/1609.05140.pdf (Original OC network, Nature CNN)
OCNET_FILTERS = [
    [32, [8, 8], 4],
    [64, [4, 4], 2],
    [64, [3, 3], 1]
]
OCNET_DENSE = 512

# A3C Atari filters as in https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17421/16660 (A2OC network)
A2OCNET_FILTERS = [
    [16, [8, 8], 4],
    [32, [4, 4], 2]
]
A2OCNET_DENSE = 256

# View layer abstraction so I can use Sequential
class View(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def forward(self, x):
        return x.view(-1, *self.sizes)  # i.e., -1, num_options, num_outputs

def _get_filter_config(shape):
    shape = list(shape)
    if len(shape) == 3 and shape[:2] == [84, 84]:
        return OCNET_FILTERS
    else:
        raise ValueError(
            "No default configuration for obs shape {}".format(shape) +
            ", you must specify `conv_filters` manually as a model option. "
            "Default configurations are only available for inputs of shape "
            "[42, 42, K] and [84, 84, K]. You may alternatively want "
            "to use a custom model or preprocessor.")

class OptionCriticVisionNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        layers = []
        (w, h, in_channels) = obs_space.shape
        in_size = [w, h]
        # Convolutional layers
        for out_channels, kernel, stride in OCNET_FILTERS:
            padding, out_size = same_padding(in_size, kernel, [stride, stride])
            layers.append(nn.Conv2d(in_channels, out_channels, kernel, stride, padding))
            layers.append(nn.ReLU())
            in_channels = out_channels
            in_size = out_size
        # Dense layer after flattening output, using ReLU
        hSize = OCNET_DENSE
        num_options = model_config.get('oc_num_options')
        self.option_epsilon = model_config.get('oc_option_epsilon')
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_size, hSize))
        layers.append(nn.ReLU())
        self._convs = nn.Sequential(*layers)
        # Value function does share layers

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = True
        self._logits = None

        # q, pi, beta, and v
        self.q = nn.Linear(hSize, num_options)  # Value for each option
        self.v = nn.Linear(hSize, 1)  # Value for state alone? Or do
        self.pi = nn.Sequential(nn.Linear(hSize, num_options * num_outputs), View((num_options, num_outputs)), nn.Softmax(dim=-1))  # Action probabilities for each option
        self.beta = nn.Sequential(nn.Linear(hSize, num_options), nn.Sigmoid)  # Termination probabilities
        # Holds the current "base" output (before heads).
        self._features = self._q = self._v = self._pi = self._beta = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._features = input_dict["obs"].float().permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        self._features = conv_out
        self._q, self._pi, self._beta = self.q(self._features), self.pi(self._features), self.beta(self._features)
        return self._pi, state  # self._pi has logits for each option, need to downselect to just correct option

    @override(TorchModelV2)
    def value_function(self):
        assert self._q is not None, "must call forward() first"
        eps = self.option_epsilon
        self._v = torch.max(self._q, dim=-1) * (1-eps) + torch.mean(self._q, dim=1) * eps
        return self._v

    def _hidden_layers(self, obs):
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        return res