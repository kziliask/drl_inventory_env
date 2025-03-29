import gymnasium as gym
from stable_baselines3 import PPO  # , SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from typing import TypedDict

gym.register(
    id="MultiInv-v0",
    entry_point="env.multibase_env:MultiInvEnv",
)
# set starting inventory low for all except one
starting_inventory = [20_000.0 for _ in range(10)]
# starting_inventory.append(100_000.0)
kwargs = {
    "n_locations": 10,
    "n_timesteps": 365,
    "starting_inventory": starting_inventory,
    "lead_times": [[2 for _ in range(10)] for _ in range(10)],
    "unit_costs": [[1.0 for _ in range(10)] for _ in range(10)],
    "unit_cost_param": 1.0,
    "mission_params": None,
    "random_demand_params": [
        {"missions": (2, 6), "distance": (500.0, 1000.0), "aircraft": (2, 4)}
        for _ in range(10)
    ],
    "random_demand_fixed_param": 1000.0,
    "random_demand_dist_params": {"p": (0.01, 0.03)},
    "seed": None,
    "scenario": "base",
    "variance": 1.0,
}


class WrapperKwargs(TypedDict):
    norm_obs: bool
    norm_reward: bool
    clip_obs: float
    clip_reward: float


wrapper_kwargs: WrapperKwargs = {
    "norm_obs": True,
    "norm_reward": True,
    "clip_obs": 10.0,
    "clip_reward": 10.0,
}
n_envs = 8
env = make_vec_env(
    "MultiInv-v0",
    n_envs=n_envs,
    monitor_dir="./ppo_csv_multiinv_tensorboard/",
    env_kwargs=kwargs,
)
env.render_mode = None
env = VecNormalize(
    env,
    **wrapper_kwargs,
    norm_obs_keys=[
        "inventory",
        "expected_demand",
        "expected_demand_ratio",
        "pipeline_inventory",
    ],
)
# Define network architecture
policy_kwargs = dict(
    net_arch=dict(
        pi=[128, 128, 128], vf=[128, 128, 128], qf=[128, 128, 128]
    )  # Separate networks for policy (pi) and value function (vf)
)
# PPO MODEL
model_ppo = PPO(
    "MultiInputPolicy",
    env,
    batch_size=256,
    gamma=0.97,
    policy_kwargs=policy_kwargs,
    verbose=1,
    clip_range=0.25,
    learning_rate=0.0001,
    tensorboard_log="./ppo_multiinv_tensorboard/",
)
model_ppo.learn(total_timesteps=10_000_000)
model_ppo.save(f"ppo_multiinv_1M_{kwargs['scenario']}")
