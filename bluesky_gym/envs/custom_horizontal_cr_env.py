import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

# Constants (keep unchanged for compatibility)
DISTANCE_MARGIN = 5 # km
REACH_REWARD = 1

DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1

NUM_INTRUDERS = 5
NUM_WAYPOINTS = 1
INTRUSION_DISTANCE = 5 # NM

WAYPOINT_DISTANCE_MIN = 100
WAYPOINT_DISTANCE_MAX = 150

D_HEADING = 45

AC_SPD = 150

NM2KM = 1.852

ACTION_FREQUENCY = 10

class HorizontalCREnv(gym.Env):
    """ 
    Horizontal Conflict Resolution Environment
    Supports stress-testing via configurable intruder speeds, headings,
    sensor noise, and variable action frequency.
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self,
                 render_mode=None,
                 intruder_speed_range=(AC_SPD, AC_SPD),
                 heading_diff_range=(45, 315),
                 sensor_noise_std=0.0,
                 action_frequency_range=(ACTION_FREQUENCY, ACTION_FREQUENCY)):
        # window setup
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)

        # parameterize distributions
        self.intruder_speed_range = intruder_speed_range
        self.heading_diff_range = heading_diff_range
        self.sensor_noise_std = sensor_noise_std
        self.action_frequency_range = action_frequency_range

        # standard observation/action spaces
        self.observation_space = spaces.Dict({
            "intruder_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "x_difference_speed": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "y_difference_speed": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "waypoint_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "cos_drift": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "sin_drift": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64)
        })
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # init Bluesky
        bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 5;FF')

        # logging
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])
        self.true_conflict = False

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        bs.traf.reset()
        # reset logs
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])
        self.true_conflict = False

        # ownship
        bs.traf.cre('KL001', actype="A320", acspd=AC_SPD)

        # dynamic intruder spawn can be added here for stress
        self._generate_conflicts()
        self._generate_waypoint()

        # sample current action frequency
        self.current_action_frequency = int(np.random.randint(*self.action_frequency_range))

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action):
        self._get_action(action)
        for _ in range(self.current_action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame()

        obs = self._get_obs()
        reward, terminated = self._get_reward()
        info = self._get_info()

        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return obs, reward, terminated, False, info

    def _generate_conflicts(self, acid='KL001'):
        target_idx = bs.traf.id2idx(acid)
        for i in range(NUM_INTRUDERS):
            dpsi = float(np.random.uniform(*self.heading_diff_range))
            cpa  = float(np.random.uniform(0, INTRUSION_DISTANCE))
            tlosh = float(np.random.uniform(100, 1000))
            speed = float(np.random.uniform(*self.intruder_speed_range))
            bs.traf.creconfs(acid=f'{i}', actype="A320",
                             targetidx=target_idx,
                             dpsi=dpsi, dcpa=cpa, tlosh=tlosh,
                             gs=speed)

    def _generate_waypoint(self, acid='KL001'):
        self.wpt_lat, self.wpt_lon, self.wpt_reach = [], [], []
        for _ in range(NUM_WAYPOINTS):
            wpt_dis = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
            ac_idx = bs.traf.id2idx(acid)
            lat, lon = fn.get_point_at_distance(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                wpt_dis, 0)
            self.wpt_lat.append(lat)
            self.wpt_lon.append(lon)
            self.wpt_reach.append(0)

    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')
        intr_ds, cos_b, sin_b, x_sp, y_sp = [],[],[],[],[]
        wpt_ds, cos_d, sin_d = [],[],[]

        self.ac_hdg = bs.traf.hdg[ac_idx]
        for i in range(NUM_INTRUDERS):
            idx = i+1
            qdr, dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                bs.traf.lat[idx], bs.traf.lon[idx])
            # noise
            dist_km = dis * NM2KM + np.random.normal(0, self.sensor_noise_std)
            intr_ds.append(dist_km)
            # bearing
            b = fn.bound_angle_positive_negative_180(self.ac_hdg - qdr)
            cos_b.append(np.cos(np.deg2rad(b)))
            sin_b.append(np.sin(np.deg2rad(b)))
            # speed diff
            hd_diff = self.ac_hdg - bs.traf.hdg[idx]
            x_sp.append(-np.cos(np.deg2rad(hd_diff)) * bs.traf.gs[idx])
            y_sp.append(bs.traf.gs[ac_idx] - np.sin(np.deg2rad(hd_diff)) * bs.traf.gs[idx])
            # update true conflict
            if dist_km < INTRUSION_DISTANCE:
                self.true_conflict = True

        for lat, lon in zip(self.wpt_lat, self.wpt_lon):
            qdr, wpd = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)
            wpt_ds.append(wpd * NM2KM)
            drift = fn.bound_angle_positive_negative_180(self.ac_hdg - qdr)
            cos_d.append(np.cos(np.deg2rad(drift)))
            sin_d.append(np.sin(np.deg2rad(drift)))
            self.average_drift = np.append(self.average_drift, abs(np.deg2rad(drift)))

        return {
            "intruder_distance": np.array(intr_ds)/WAYPOINT_DISTANCE_MAX,
            "cos_difference_pos": np.array(cos_b),
            "sin_difference_pos": np.array(sin_b),
            "x_difference_speed": np.array(x_sp)/AC_SPD,
            "y_difference_speed": np.array(y_sp)/AC_SPD,
            "waypoint_distance": np.array(wpt_ds)/WAYPOINT_DISTANCE_MAX,
            "cos_drift": np.array(cos_d),
            "sin_drift": np.array(sin_d)
        }

    def _get_info(self):
        return {
            'total_reward': self.total_reward,
            'total_intrusions': self.total_intrusions,
            'average_drift': self.average_drift.mean(),
            'true_conflict': int(self.true_conflict)
        }

    def _get_reward(self):
        reach_r = self._check_waypoint()
        drift_r = self._check_drift()
        intr_r  = self._check_intrusion()
        total = reach_r + drift_r + intr_r
        self.total_reward += total
        done = 0 if 0 in self.wpt_reach else 1
        return total, done

    def _check_waypoint(self):
        r=0
        for i,dist in enumerate(self._get_obs()["waypoint_distance"]*WAYPOINT_DISTANCE_MAX):
            if dist < DISTANCE_MARGIN and not self.wpt_reach[i]:
                self.wpt_reach[i]=1; r+=REACH_REWARD
        return r

    def _check_drift(self):
        return self.average_drift[-1] * DRIFT_PENALTY

    def _check_intrusion(self):
        r=0
        ac_idx = bs.traf.id2idx('KL001')
        for i in range(NUM_INTRUDERS):
            _, d = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                bs.traf.lat[i+1], bs.traf.lon[i+1])
            if d < INTRUSION_DISTANCE:
                self.total_intrusions+=1; r+=INTRUSION_PENALTY
        return r

    def _get_action(self, action):
        hdg = self.ac_hdg + action[0] * D_HEADING
        bs.stack.stack(f"HDG KL001 {hdg}")

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 200 # width of screen in km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2-heading_end_x/2,self.window_height/2+heading_end_y/2),
            ((self.window_width/2)+heading_end_x/2,(self.window_height/2)-heading_end_y/2),
            width = 4
        )

        # draw heading line
        heading_length = 50
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2,self.window_height/2),
            ((self.window_width/2)+heading_end_x,(self.window_height/2)-heading_end_y),
            width = 1
        )

        # draw intruders
        ac_length = 3

        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # determine color
            if int_dis < INTRUSION_DISTANCE:
                color = (220,20,60)
            else: 
                color = (80,80,80)

            x_pos = (self.window_width/2)+(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_width
            y_pos = (self.window_height/2)-(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # draw heading line
            heading_length = 10
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
            )

            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.window_width,
                width = 2
            )

            # import code
            # code.interact(local=locals())

        # draw target waypoint
        for qdr, dis, reach in zip(self.wpt_qdr, self.waypoint_distance, self.wpt_reach):

            circle_x = ((np.cos(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width
            circle_y = ((np.sin(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width

            if reach:
                color = (155,155,155)
            else:
                color = (255,255,255)

            pygame.draw.circle(
                canvas, 
                color,
                ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
                radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
                width = 2
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        pass
