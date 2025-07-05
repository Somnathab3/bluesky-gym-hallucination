# marl_conflict_env.py

import numpy as np
import pygame
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from typing import Dict, List, Any, Optional

# -----------------------------------------------------------------------------
# 1) Constants & conversion factors (matching the PDF examples)
# -----------------------------------------------------------------------------
NM2KM = 1.852
INTRUSION_DISTANCE_NM = 5
VERTICAL_MARGIN_FT = 1000
ACTION_FREQ = 5            # how many bs.sim.step() per agent step
NUM_NEARBY = 4             # match SectorCREnv’s NUM_AC_STATE

# -----------------------------------------------------------------------------
# 2) Environment class
# -----------------------------------------------------------------------------
class MARLConflictResolutionEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        num_aircraft: int = 6,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        cooperative_reward: bool = True,
    ):
        super().__init__()

        # parameters
        self.num_aircraft = num_aircraft
        self.render_mode   = render_mode
        self.max_steps     = max_episode_steps
        self.coop_reward   = cooperative_reward

        # --- BlueSky init (once) ---
        bs.init(mode="sim", detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack("DT 1;FF")

        # agents
        self.possible_agents = [f"AC{i}" for i in range(num_aircraft)]
        self.agents = self.possible_agents.copy()
        self.step_count = 0

        # spaces (exactly as in the PDF’s SectorCREnv example)
        self.observation_space = spaces.Dict({
            "cos(drift)": spaces.Box(-1,1,shape=(1,),dtype=np.float32),
            "sin(drift)": spaces.Box(-1,1,shape=(1,),dtype=np.float32),
            "airspeed":   spaces.Box(  0,1,shape=(1,),dtype=np.float32),
            "x_r":        spaces.Box(-np.inf,np.inf,shape=(NUM_NEARBY,),dtype=np.float32),
            "y_r":        spaces.Box(-np.inf,np.inf,shape=(NUM_NEARBY,),dtype=np.float32),
            "vx_r":       spaces.Box(-np.inf,np.inf,shape=(NUM_NEARBY,),dtype=np.float32),
            "vy_r":       spaces.Box(-np.inf,np.inf,shape=(NUM_NEARBY,),dtype=np.float32),
            "cos(track)": spaces.Box(-1,1,shape=(NUM_NEARBY,),dtype=np.float32),
            "sin(track)": spaces.Box(-1,1,shape=(NUM_NEARBY,),dtype=np.float32),
            "distances":  spaces.Box(  0,np.inf,shape=(NUM_NEARBY,),dtype=np.float32),
        })
        self.action_space = spaces.Box(-1,1,shape=(2,),dtype=np.float32)

        # apply to all
        self.observation_spaces = {a:self.observation_space for a in self.possible_agents}
        self.action_spaces      = {a:self.action_space      for a in self.possible_agents}

        # rendering fields
        self.window = None
        self.clock  = None
        self.w, self.h = 512, 512

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        # reset simulator
        bs.traf.reset()
        self.step_count = 0
        self.agents = self.possible_agents.copy()
        # create our aircraft
        self._spawn_aircraft()
        # build initial obs/infos
        obs = self._get_observations()
        infos = {a:{} for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str,np.ndarray]):
        self.step_count += 1
        # apply each agent’s action
        for a, act in actions.items():
            self._apply_action(a, act)
        # advance the sim
        for _ in range(ACTION_FREQ):
            bs.sim.step()
        # observations, rewards, done flags
        obs    = self._get_observations()
        rewards= self._calc_rewards()
        term   = {a:False for a in self.agents}
        trunc  = {a:(self.step_count>=self.max_steps) for a in self.agents}
        infos  = {a:{} for a in self.agents}
        return obs, rewards, term, trunc, infos

    def render(self, mode="human"):
        if mode=="rgb_array":
            return self._draw_frame()
        elif mode=="human":
            self._draw_frame()

    # --------------- internal helpers ---------------

    def _spawn_aircraft(self):
        """Place each AC at random lat/lon and heading."""
        self._ac_indices = {}
        for i, acid in enumerate(self.possible_agents):
            # simple random box
            lat = np.random.uniform(-1,1)
            lon = np.random.uniform(-1,1)
            hdg = np.random.uniform(0,360)
            bs.traf.cre(acid, actype="A320",
                        aclat=lat, aclon=lon,
                        achdg=hdg, acspd=150, acalt=30000)
            idx = bs.traf.id2idx(acid)
            self._ac_indices[acid] = idx

    def _apply_action(self, acid:str, action:np.ndarray):
        """Map [-1,1]^2 → heading & speed changes."""
        idx = self._ac_indices[acid]
        dh = action[0]*22.5   # ±22.5°
        dv = action[1]*20     # ±20 kt
        new_hdg = fn.bound_angle_positive_negative_180(bs.traf.hdg[idx]+dh)
        new_spd = np.clip(bs.traf.cas[idx]+dv, 100, 300)
        bs.stack.stack(f"HDG {acid} {new_hdg}")
        bs.stack.stack(f"SPD {acid} {new_spd}")

    def _get_observations(self):
        out = {}
        # find center AC for drift calc
        center_ac = self.possible_agents[0]
        cidx = self._ac_indices[center_ac]
        # compute drift to next waypoint = 0 here
        drift = 0.0
        cosd, sind = np.cos(np.deg2rad(drift)), np.sin(np.deg2rad(drift))
        for acid in self.agents:
            idx = self._ac_indices[acid]
            # build SectorCREnv–style obs
            v = bs.traf.tas[idx]/500.0
            # find nearest NUM_NEARBY intruders
            dlist=[]
            for other in self.agents:
                if other==acid: continue
                j = self._ac_indices[other]
                qdr, dist = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    bs.traf.lat[j],   bs.traf.lon[j])
                dlist.append((other,dist,qdr,j))
            dlist.sort(key=lambda x:x[1])
            # fill arrays
            x_r = np.zeros(NUM_NEARBY)
            y_r = np.zeros(NUM_NEARBY)
            vx_r= np.zeros(NUM_NEARBY)
            vy_r= np.zeros(NUM_NEARBY)
            ctrk= np.zeros(NUM_NEARBY)
            strk= np.zeros(NUM_NEARBY)
            dists=np.zeros(NUM_NEARBY)
            for k,(_,dist,qdr,j) in enumerate(dlist[:NUM_NEARBY]):
                # relative pos
                rad = np.deg2rad(qdr)
                x_r[k] =  dist*np.cos(rad)/50_000
                y_r[k] =  dist*np.sin(rad)/50_000
                # relative vel
                dvx = (bs.traf.tas[j]*np.cos(np.deg2rad(bs.traf.hdg[j])) 
                      - bs.traf.tas[idx]*np.cos(np.deg2rad(bs.traf.hdg[idx])))
                dvy = (bs.traf.tas[j]*np.sin(np.deg2rad(bs.traf.hdg[j])) 
                      - bs.traf.tas[idx]*np.sin(np.deg2rad(bs.traf.hdg[idx])))
                vx_r[k], vy_r[k] = dvx/100, dvy/100
                # track
                trk = np.arctan2(dvy,dvx)
                ctrk[k]=np.cos(trk)
                strk[k]=np.sin(trk)
                dists[k]=dist/100
            out[acid] = {
                "cos(drift)": np.array([cosd],dtype=np.float32),
                "sin(drift)": np.array([sind],dtype=np.float32),
                "airspeed":   np.array([v],dtype=np.float32),
                "x_r":        x_r.astype(np.float32),
                "y_r":        y_r.astype(np.float32),
                "vx_r":       vx_r.astype(np.float32),
                "vy_r":       vy_r.astype(np.float32),
                "cos(track)": ctrk.astype(np.float32),
                "sin(track)": strk.astype(np.float32),
                "distances":  dists.astype(np.float32),
            }
        return out

    def _calc_rewards(self)->Dict[str,float]:
        # very simple: penalize intrusions
        r = {}
        for acid in self.agents:
            idx = self._ac_indices[acid]
            pen = 0
            for other in self.agents:
                if other==acid: continue
                j = self._ac_indices[other]
                _, dist = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    bs.traf.lat[j],   bs.traf.lon[j])
                if dist<INTRUSION_DISTANCE_NM:
                    pen -=1
            r[acid]=float(pen)
        return r

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()

    def _draw_frame(self):
        # very minimal rgb_array rendering
        surf = pygame.Surface((self.w,self.h))
        surf.fill((135,206,235))
        pg = pygame.surfarray.pixels3d(surf)
        return np.transpose(pg, (1,0,2)).copy()

# -----------------------------------------------------------------------------
# 3) Gym registration
# -----------------------------------------------------------------------------
register(
    id="MARLConflictResolution-v0",
    entry_point="bluesky_gym_marl.marl_conflict_env:MARLConflictResolutionEnv",
)
