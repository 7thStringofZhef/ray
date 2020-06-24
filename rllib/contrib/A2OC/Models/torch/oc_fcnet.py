from .utils import View
from .oc_cfg import *

import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, try_import_torch
torch, nn = try_import_torch()

class OptionCriticFullyConnectedNetwork(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        num_options = model_config.get('oc_num_options')
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs * num_options,
                              model_config, name)
        nn.Module.__init__(self)

        activation = get_activation_fn(
            model_config.get("fcnet_activation"), framework="torch")
        hiddens = model_config.get("fcnet_hiddens")
        self.option_epsilon = model_config.get('oc_option_epsilon')

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))

        # Create layers
        for size in hiddens:
            layers.append(nn.Linear(prev_layer_size, size))
            layers.append(activation)
            prev_layer_size = size
        self._body = nn.Sequential(*layers)
        self.q = nn.Linear(prev_layer_size, num_options)  # Value for each option
        self.pi = nn.Sequential(nn.Linear(prev_layer_size, num_options * num_outputs), View((num_options, num_outputs)), nn.Softmax(dim=-1))  # Action probabilities for each option
        self.beta = nn.Sequential(nn.Linear(prev_layer_size, num_options), nn.Sigmoid)  # Termination probabilities
        # Holds the current "base" output (before logits layer).
        self._features = self._q = self._v = self._pi = self._beta = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._body(self._last_flat_in)
        self._q, self._pi, self._beta = self.q(self._features), self.pi(self._features), self.beta(self._features)
        return self._pi.view(self._q.size(0), -1), state  # self._pi has logits for each option, need to downselect to just correct option in the policy

    @override(TorchModelV2)
    def value_function(self):
        assert self._q is not None, "must call forward() first"
        eps = self.option_epsilon
        self._v = torch.max(self._q, dim=-1) * (1-eps) + torch.mean(self._q, dim=1) * eps
        return self._v
