import ray
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

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
    model = OptionCriticVisionNetwork(obs_space, action_space, action_space.n, config, 'test')
    dist_class = None
    return model, dist_class

# Should need either this or the action distribution function
def option_critic_action_sampler_fn(policy, model, input_dict, obs_space, action_space, config):
    action = action_logp = None
    return action, action_logp

def option_critic_action_distribution_fn(policy, model, train_batch, explore, t, training):
    dist_inputs = dist_class = state = None
    return dist_inputs, dist_class, state

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