# bluesky_gym/__init__.py
# Main BlueSky-Gym registration file

from gymnasium.envs.registration import register

def register_envs():
    """
    Import this module so that environments / scenarios register themselves.
    """
    
    # Register existing environments
    register(
        id="DescentEnv-v0",
        entry_point="bluesky_gym.envs.descent_env:DescentEnv",
        max_episode_steps=300,
    )
    
    register(
        id="PlanWaypointEnv-v0",
        entry_point="bluesky_gym.envs.plan_waypoint_env:PlanWaypointEnv",
        max_episode_steps=300,
    )
    
    register(
        id="HorizontalCREnv-v0",
        entry_point="bluesky_gym.envs.horizontal_cr_env:HorizontalCREnv",
        max_episode_steps=300,
    )
    
    register(
        id="VerticalCREnv-v0",
        entry_point="bluesky_gym.envs.vertical_cr_env:VerticalCREnv",
        max_episode_steps=300,
    )
    
    register(
        id="SectorCREnv-v0",
        entry_point="bluesky_gym.envs.sector_cr_env:SectorCREnv",
        max_episode_steps=300,
    )
    
    register(
        id="StaticObstacleEnv-v0",
        entry_point="bluesky_gym.envs.static_obstacle_env:StaticObstacleEnv",
        max_episode_steps=300,
    )
    
    register(
        id="MergeEnv-v0",
        entry_point="bluesky_gym.envs.merge_env:MergeEnv",
        max_episode_steps=300,
    )
    
    # Register the new custom conflict resolution environments
    register(
        id="CustomHorizontalCREnv-v0",
        entry_point="bluesky_gym.envs.custom_horizontal_cr_env:CustomHorizontalCREnv",
        max_episode_steps=500,
        reward_threshold=10.0,
    )
    
    register(
        id="CustomVerticalCREnv-v0", 
        entry_point="bluesky_gym.envs.custom_vertical_cr_env:CustomVerticalCREnv",
        max_episode_steps=400,
        reward_threshold=8.0,
    )
