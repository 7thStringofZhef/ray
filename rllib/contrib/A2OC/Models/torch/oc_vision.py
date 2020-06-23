import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, same_padding, \
    SlimConv2d, SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, try_import_torch

import torch.nn as nn
#_, nn = try_import_torch()

# Atari filters as in https://arxiv.org/pdf/1609.05140.pdf
OCNET_FILTERS = [
    [32, [8, 8], 4],
    [64, [4, 4], 2],
    [64, [3, 3], 1]
]
OCNET_DENSE = 512

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
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_size, OCNET_DENSE))
        layers.append(nn.ReLU())
        self._convs = nn.Sequential(*layers)
        vf_share_layers = model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = True
        self._logits = None


        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        self._value_branch = SlimFC(
            out_channels, 1, initializer=normc_initializer(0.01))

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._features = input_dict["obs"].float().permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            logits = conv_out.squeeze(3)
            logits = logits.squeeze(2)
            return logits, state
        else:
            return conv_out, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            value = self._value_branch_separate(self._features)
            value = value.squeeze(3)
            value = value.squeeze(2)
            return value.squeeze(1)
        else:
            if not self.last_layer_is_flattened:
                features = self._features.squeeze(3)
                features = features.squeeze(2)
            else:
                features = self._features
            return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs):
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res