from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import same_padding
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from .oc_cfg import *
from .utils import View

torch, nn = try_import_torch()

class OptionCriticVisionNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        num_options = model_config.get('oc_num_options')
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs * num_options,
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
        self.option_epsilon = model_config.get('oc_option_epsilon')
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_size, hSize))
        layers.append(nn.ReLU())
        self._convs = nn.Sequential(*layers)
        # q, pi, beta, and v
        self.q = nn.Linear(hSize, num_options)  # Value for each option
        #self.v = nn.Linear(hSize, 1)  # Value for state alone? Or do
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
        return self._pi.view(self._q.size(0), -1), state  # self._pi has logits for each option, need to downselect to just correct option in the policy

    @override(TorchModelV2)
    def value_function(self):
        assert self._q is not None, "must call forward() first"
        eps = self.option_epsilon
        self._v = torch.max(self._q, dim=-1) * (1-eps) + torch.mean(self._q, dim=1) * eps
        return self._v

    def _hidden_layers(self, obs):
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        return res