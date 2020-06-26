import ray
import numpy as np
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, convert_to_torch_tensor

from ..Models.torch.oc_vision import OptionCriticVisionNetwork

import torch
import torch.nn as nn
#_, nn = try_import_torch()

# Option critic loss function
def option_critic_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    overall_err = 0
    return overall_err

# Postprocessing for an option-critic trajectory
# Probably compute advantages in here
def option_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    return {SampleBatch.VF_PREDS: 0}

# Extra action outputs. Save things like value, option, etc
def option_critic_extra_action_out(policy, input_dict, state_batches, model, action_dist):
    pass

# Stats for policy (entropy, policy loss, termination loss, value loss)
def option_critic_stats(policy, train_batch):
    pass

# Default optimizer (RMSProp)
def option_critic_optimizer(policy, config):
    return torch.optim.rmsprop.RMSprop(policy.model.parameters(), lr=config["lr"])

# Gradient clipping
def option_critic_gradient_process(policy, optimizer, loss):
    info = {}
    if policy.config["grad_clip"]:
        for param_group in optimizer.param_groups:
            params = list(
                filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                grad_gnorm = nn.utils.clip_grad_norm_(
                    params, policy.config["grad_clip"])
                if isinstance(grad_gnorm, torch.Tensor):
                    grad_gnorm = grad_gnorm.cpu().numpy()
                info["grad_gnorm"] = grad_gnorm
    return info

# make the model appropriate to the observation and action space
def option_critic_make_model_and_action_dist(policy, obs_space, action_space, config):
    # Basic distribution class should be fine as long as I input the logits corresponding to the correct option
    dist_class = ModelCatalog.get_action_dist(
        action_space,
        config,
        framework="torch"
    )
    # option critic vision network. May want to revise to register this as a custom model, then grab it
    model = OptionCriticVisionNetwork(
        obs_space,
        action_space,
        action_space.n,
        config, 
        'test')
    return model, dist_class

# Sample actions
def option_critic_action_sampler_fn(policy, model, input_dict, obs_space, action_space, config):
    action = action_logp = None
    return action, action_logp

# After rest of policy is setup, start with initial option and delib cost
def option_critic_after_init(policy, observation_space, action_space, config):
    pass

A2OCTorchPolicy = build_torch_policy(
    name="A2OCTorchPolicy",
    get_default_config=lambda: ray.rllib.agents.a3c.a3c.DEFAULT_CONFIG,
    loss_fn=option_critic_loss,
    stats_fn=option_critic_stats,
    extra_action_out_fn=option_critic_extra_action_out,
    extra_grad_process_fn=option_critic_gradient_process,
    optimizer_fn=option_critic_optimizer,
    make_model_and_action_dist=option_critic_make_model_and_action_dist,
)

class A2OCTorchPolicyClass(TorchPolicy):
    def __init__(self, obs_space, action_space, config, model,
                 loss=option_critic_loss,
                 action_distribution_class=None,
                 action_sampler_fn=option_critic_action_sampler_fn):
        super().__init__(obs_space,
                         action_space,
                         config,
                         model=model,
                         loss=loss,
                         action_distribution_class=action_distribution_class,
                         action_sampler_fn=action_sampler_fn)
        self.current_o = 0
        self.prev_o = 0

    @override(TorchPolicy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        # Exploration class will take action dist, timestep, and explore, return torch/tf tensor action
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        with torch.no_grad():
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
            input_dict = self._lazy_tensor_dict({
                SampleBatch.CUR_OBS: obs_batch,
                "is_training": False,
            })
            if prev_action_batch is not None:
                input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
            if prev_reward_batch is not None:
                input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
            state_batches = [
                self._convert_to_tensor(s) for s in (state_batches or [])
            ]
            # Use action_sampler_fn variant
            action_dist = dist_inputs = None
            state_out = []
            actions, logp = self.action_sampler_fn(
                self,
                self.model,
                input_dict,
                explore=explore,
                timestep=timestep)

            input_dict[SampleBatch.ACTIONS] = actions

            # Add default and custom fetches.
            extra_fetches = self.extra_action_out(input_dict, state_batches,
                                                  self.model, action_dist)
            # Action-logp and action-prob.
            if logp is not None:
                logp = convert_to_non_torch_type(logp)
                extra_fetches[SampleBatch.ACTION_PROB] = np.exp(logp)
                extra_fetches[SampleBatch.ACTION_LOGP] = logp
            # Action-dist inputs.
            if dist_inputs is not None:
                extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs
            return convert_to_non_torch_type((actions, state_out,
                                              extra_fetches))

    @override(TorchPolicy)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        pass

