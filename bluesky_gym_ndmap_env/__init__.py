## bluesky_gym_ndmap_env/__init__.py

import gymnasium as gym
from gymnasium.envs.registration import register

# Register the ND-Map conflict resolution environment
register(
    id="NDMapConflictEnv-v0",
    entry_point="bluesky_gym_ndmap_env.ndmap_conflict_env:NDMapConflictEnv",
    max_episode_steps=1000,
    kwargs={
        "airspace_bounds": (0.0, 10.0, 0.0, 10.0),  # lat_min, lat_max, lon_min, lon_max (degrees)
        "altitude_bounds": (20000, 40000),  # min_alt, max_alt (feet)
        "time_horizon": 3600.0,  # 1 hour simulation (seconds)
        "separation_lateral_nm": 5.0,  # 5 NM lateral separation
        "separation_vertical_ft": 1000.0,  # 1000 ft vertical separation
        "max_intruders": 10,
        "tile_arity": 16,  # ND-Map tile arity as per Kuenz dissertation
    }
)
