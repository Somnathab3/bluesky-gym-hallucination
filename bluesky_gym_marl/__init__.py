# bluesky_gym_marl/__init__.py

"""
BlueSky Gym MARL - Multi-Agent Reinforcement Learning for Air Traffic Conflict Resolution
"""

import gymnasium as gym
from gymnasium.envs.registration import register

# Import the environment to make it available
from .marl_conflict_env import MARLConflictEnv
from .utils import (
    make_sb3_env, 
    make_individual_agent_env, 
    RewardNormalizer,
    evaluate_marl_performance,
    group_agents_by_sector,
    MALRLLogger
)

# Register the environment with Gymnasium
register(
    id="MARLConflict-v0",
    entry_point="bluesky_gym_marl.marl_conflict_env:MARLConflictEnv",
    kwargs={
        "num_aircraft": 8,
        "airspace_size": 20.0,
        "separation_lateral_nm": 5.0,
        "separation_vertical_ft": 1000.0,
        "max_episode_steps": 2000,
        "cooperative_reward": True,
        "sector_based": False,
        "communication_enabled": False,
        "observation_radius_nm": 15.0
    }
)

# Register variants for different scenarios
register(
    id="MARLConflict-Cooperative-v0",
    entry_point="bluesky_gym_marl.marl_conflict_env:MARLConflictEnv",
    kwargs={
        "num_aircraft": 12,
        "cooperative_reward": True,
        "communication_enabled": True,
        "sector_based": True
    }
)

register(
    id="MARLConflict-Dense-v0",
    entry_point="bluesky_gym_marl.marl_conflict_env:MARLConflictEnv",
    kwargs={
        "num_aircraft": 16,
        "airspace_size": 15.0,
        "separation_lateral_nm": 4.0,
        "max_episode_steps": 3000
    }
)

register(
    id="MARLConflict-Simple-v0",
    entry_point="bluesky_gym_marl.marl_conflict_env:MARLConflictEnv",
    kwargs={
        "num_aircraft": 4,
        "airspace_size": 10.0,
        "max_episode_steps": 1000,
        "cooperative_reward": False
    }
)

__version__ = "0.1.0"
__all__ = [
    "MARLConflictEnv",
    "make_sb3_env",
    "make_individual_agent_env", 
    "RewardNormalizer",
    "evaluate_marl_performance",
    "group_agents_by_sector",
    "MALRLLogger"
]
