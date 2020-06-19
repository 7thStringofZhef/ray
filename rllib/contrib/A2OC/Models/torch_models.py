from abc import ABC
import numpy as np

from ray.rllib.models.model import restore_original_dimensions, flatten
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, AppendBiasLayer
from ray.rllib.utils.annotations import override
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.framework import try_import_torch

import torch
import torch.nn as nn
# torch, nn = try_import_torch()

def convert_to_tensor(arr):
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor

class OptionCriticModel(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.preprocessor = get_preprocessor(obs_space.original_space)(
            obs_space.original_space)

        self.body = None  # Shared body for all 3 heads of OC
        self.q = None
        self.beta = None
        self.pi = None

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        x = self.body(x)
        # actor outputs
        logits = self.actor_layers(x)

        # compute value
        self._value_out = self.critic_layers(x)
        return logits, None

    def value_function(self):
        return self._value_out

    def compute_priors_and_value(self, obs):
        obs = convert_to_tensor([self.preprocessor.transform(obs)])
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvNetModel(OptionCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        OptionCriticModel.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)

        in_channels = model_config["custom_model_config"]["in_channels"]
        feature_dim = model_config["custom_model_config"]["feature_dim"]

        self.shared_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.Conv2d(64, 64, kernel_size=2, stride=1), Flatten(),
            nn.Linear(1024, feature_dim))

        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=action_space.n))

        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=1))

        self._value_out = None


class DenseModel(OptionCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        OptionCriticModel.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)

        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features=obs_space.original_space["obs"].shape[0],
                out_features=256), nn.Linear(
                    in_features=256, out_features=256))
        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=action_space.n))
        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        self._value_out = None

class HardCodedOptionCriticModel(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        activation = nn.ReLU
        filters = model_config.get("conv_filters")
        filters = [
                    [16, [2, 2], 1],
                    [16, [2, 2], 1],
                    [32, [2, 2], 1]
                    ]
        # no_final_linear = model_config.get("no_final_linear")
        # vf_share_layers = model_config.get("vf_share_layers")

        layers = []
        (w, h, in_channels) = obs_space.shape
        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = valid_padding(in_size, kernel,
                                              [stride, stride])
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation))
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        layers.append(
            SlimConv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                None,
                activation_fn=activation))
        self._convs = nn.Sequential(*layers)

        self._logits = SlimFC(
            out_channels, num_outputs, initializer=nn.init.xavier_uniform_)
        self._value_branch = SlimFC(
            out_channels, 1, initializer=normc_initializer())
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._features = self._hidden_layers(input_dict["obs"].float())
        logits = self._logits(self._features)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)

    def _hidden_layers(self, obs):
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res