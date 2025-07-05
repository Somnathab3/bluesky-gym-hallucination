# bluesky_gym/envs/__init__.py

from .descent_env import *
from .plan_waypoint_env import *
from .static_obstacle_env import *
from .vertical_cr_env import *
from .horizontal_cr_env import *
from .merge_env import *
from .sector_cr_env import *
from .marl_conflict_env import MARLConflictResolutionEnv  # import your class by name

# register all of the built-in BSG envs
from .descent_env import register as _r1; _r1()
from .plan_waypoint_env import register as _r2; _r2()
from .static_obstacle_env import register as _r3; _r3()
from .vertical_cr_env import register as _r4; _r4()
from .horizontal_cr_env import register as _r5; _r5()
from .merge_env import register as _r6; _r6()
from .sector_cr_env import register as _r7; _r7()

# now register your MARLConflictResolutionEnv
from gymnasium.envs.registration import register
register(
    id="MARLConflictResolution-v0",
    entry_point="bluesky_gym.envs.marl_conflict_env:MARLConflictResolutionEnv",
)
