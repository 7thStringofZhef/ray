import gym
import gym_minigrid

"""Example of using training on MiniGrid-FourRooms-v0."""

import argparse

import ray
from ray import tune
from ray.rllib.models.catalog import ModelCatalog

env = gym_minigrid.wrappers.ImgObsWrapper(gym.make('MiniGrid-FourRooms-v0'))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=6, type=int)
    parser.add_argument("--training-iteration", default=10000, type=int)
    parser.add_argument("--ray-num-cpus", default=7, type=int)
    args = parser.parse_args()
    ray.init(num_cpus=args.ray_num_cpus)