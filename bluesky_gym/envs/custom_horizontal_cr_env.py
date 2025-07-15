import numpy as np
import pygame
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import json
import time
import os

# Constants
DISTANCE_MARGIN = 5  # km
REACH_REWARD = 1
DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1
NUM_WAYPOINTS = 1
INTRUSION_DISTANCE = 5  # NM
WAYPOINT_DISTANCE_MIN = 100
WAYPOINT_DISTANCE_MAX = 150
D_HEADING = 45
AC_SPD = 150
NM2KM = 1.852
ACTION_FREQUENCY = 10

class CustomHorizontalCREnv(gym.Env):
    """
    Custom Horizontal Conflict Resolution Environment for Hallucination Research
    
    Key Features:
    - Dynamic intruder count based on traffic density
    - Multiple aircraft types with different performance characteristics
    - Stress testing parameters for hallucination detection
    - Training envelope boundary detection
    - Communication delay simulation
    - Comprehensive logging for safety margin analysis
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 120}

    def __init__(
        self,
        render_mode=None,
        # Basic parameters
        num_intruders=5,
        intruder_speed_range=(AC_SPD, AC_SPD),
        heading_diff_range=(45, 315),
        sensor_noise_std=0.0,
        action_frequency_range=(ACTION_FREQUENCY, ACTION_FREQUENCY),
        
        # Hallucination research parameters
        aircraft_type="A320",
        weather_conditions="clear",
        traffic_density_factor=1.0,
        communication_delay_range=(0, 0),
        sensor_failure_prob=0.0,
        uncertainty_injection=False,
        hallucination_detection=True,
        track_training_envelope=True,
        envelope_violation_threshold=0.8,
    ):
        # Window setup
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)
        
        # Dynamic intruder configuration
        self.base_num_intruders = num_intruders
        self.traffic_density_factor = traffic_density_factor
        self.num_intruders = max(1, int(self.base_num_intruders * self.traffic_density_factor))
        
        # Stress testing parameters
        self.intruder_speed_range = intruder_speed_range
        self.heading_diff_range = heading_diff_range
        self.sensor_noise_std = sensor_noise_std
        self.action_frequency_range = action_frequency_range
        
        # Aircraft and environment configuration
        self.aircraft_type = aircraft_type
        self.aircraft_performance = self._get_aircraft_performance(aircraft_type)
        self.weather_conditions = weather_conditions
        self.communication_delay_range = communication_delay_range
        self.sensor_failure_prob = sensor_failure_prob
        
        # Hallucination research configuration
        self.uncertainty_injection = uncertainty_injection
        self.hallucination_detection = hallucination_detection
        self.track_training_envelope = track_training_envelope
        self.envelope_violation_threshold = envelope_violation_threshold
        
        # State tracking
        self.action_history = deque(maxlen=50)
        self.prediction_confidence_history = deque(maxlen=50)
        self.envelope_violations = []
        self.uncertainty_metrics = {}
        self.safety_margins = {}
        self._action_buffer = deque()
        
        # Training envelope boundaries
        self.training_envelope = {
            'speed_bounds': [100, 200],
            'heading_bounds': [0, 360], 
            'distance_bounds': [50, 200],
            'complexity_bounds': [0.1, 0.9]
        }
        
        # Observation space
        base_obs_space = {
            "intruder_distance": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64),
            "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64),
            "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64),
            "x_difference_speed": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64),
            "y_difference_speed": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64),
            "waypoint_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "cos_drift": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "sin_drift": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
        }
        
        if self.uncertainty_injection:
            base_obs_space.update({
                "observation_uncertainty": spaces.Box(0, 1, shape=(1,), dtype=np.float64),
                "environment_complexity": spaces.Box(0, 1, shape=(1,), dtype=np.float64),
                "training_envelope_distance": spaces.Box(0, np.inf, shape=(1,), dtype=np.float64),
            })
        
        self.observation_space = spaces.Dict(base_obs_space)
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Initialize BlueSky
        bs.init(mode="sim", detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack("DT 5;FF")
        
        # Episode tracking
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])
        self.true_conflict = False
        self.episode_data = {}
        self.timestep = 0
        
        # GUI elements
        self.window = None
        self.clock = None

    def _get_aircraft_performance(self, aircraft_type):
        """Get aircraft-specific performance characteristics"""
        aircraft_db = {
            "A320": {
                "max_speed": 180, "min_speed": 120, "turn_rate": 3.0,
                "mass": 78000, "wing_span": 35.8, "length": 37.6
            },
            "B747": {
                "max_speed": 250, "min_speed": 140, "turn_rate": 1.5,
                "mass": 412000, "wing_span": 68.4, "length": 70.7
            },
            "A380": {
                "max_speed": 260, "min_speed": 150, "turn_rate": 1.2,
                "mass": 575000, "wing_span": 79.8, "length": 72.7
            },
            "E190": {
                "max_speed": 170, "min_speed": 110, "turn_rate": 4.0,
                "mass": 51800, "wing_span": 28.7, "length": 36.2
            }
        }
        return aircraft_db.get(aircraft_type, aircraft_db["A320"])

    def _calculate_environment_complexity(self):
        """Calculate current environment complexity for envelope tracking"""
        if len(bs.traf.id) == 0:
            return 0.0
            
        speed_variation = np.std([bs.traf.gs[i] for i in range(len(bs.traf.id))])
        heading_variation = np.std([bs.traf.hdg[i] for i in range(len(bs.traf.id))])
        density_factor = len(bs.traf.id) / 10.0
        
        noise_factor = self.sensor_noise_std
        weather_factor = {"clear": 0.0, "turbulent": 0.3, "windy": 0.2}[self.weather_conditions]
        
        complexity = np.clip(
            (speed_variation/50 + heading_variation/180 + density_factor + 
             noise_factor + weather_factor) / 5.0, 0, 1
        )
        return complexity

    def _detect_envelope_violation(self, obs):
        """Detect if current observation is outside training envelope"""
        violations = []
        
        # Check speed envelope
        current_speeds = obs['x_difference_speed']**2 + obs['y_difference_speed']**2
        speed_violation = np.any(
            (current_speeds < (self.training_envelope['speed_bounds'][0]/AC_SPD)**2) |
            (current_speeds > (self.training_envelope['speed_bounds'][1]/AC_SPD)**2)
        )
        if speed_violation:
            violations.append('speed')
        
        # Check distance envelope
        distances = obs['intruder_distance'] * WAYPOINT_DISTANCE_MAX
        distance_violation = np.any(
            (distances < self.training_envelope['distance_bounds'][0]) |
            (distances > self.training_envelope['distance_bounds'][1])
        )
        if distance_violation:
            violations.append('distance')
        
        # Check complexity envelope
        complexity = self._calculate_environment_complexity()
        complexity_violation = (
            complexity < self.training_envelope['complexity_bounds'][0] or
            complexity > self.training_envelope['complexity_bounds'][1]
        )
        if complexity_violation:
            violations.append('complexity')
        
        return violations, len(violations) / 3.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        bs.traf.reset()
        
        # Reset state variables
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])
        self.true_conflict = False
        self.timestep = 0
        self.envelope_violations = []
        self._action_buffer.clear()
        
        # Initialize episode data
        self.episode_data = {
            'start_time': time.time(),
            'aircraft_type': self.aircraft_type,
            'weather_conditions': self.weather_conditions,
            'traffic_density_factor': self.traffic_density_factor,
            'num_intruders': self.num_intruders,
            'stress_parameters': {
                'speed_range': self.intruder_speed_range,
                'heading_range': self.heading_diff_range,
                'noise_std': self.sensor_noise_std,
                'action_freq': self.action_frequency_range,
                'sensor_failure_prob': self.sensor_failure_prob,
                'comm_delay': self.communication_delay_range
            }
        }
        
        # Create ownship
        bs.traf.cre("OWN001", actype=self.aircraft_type, 
                   acspd=self.aircraft_performance['min_speed'])
        
        # Generate conflicts and waypoint
        self._generate_conflicts()
        self._generate_waypoint()
        
        # Set action frequency
        low, high = self.action_frequency_range
        if high <= low:
            self.current_action_frequency = int(low)
        else:
            self.current_action_frequency = int(np.random.randint(low, high + 1))
        
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return obs, info

    def _generate_conflicts(self, acid="OWN001"):
        """Generate conflicts with stress testing parameters"""
        target_idx = bs.traf.id2idx(acid)
        
        for i in range(self.num_intruders):
            dpsi = float(np.random.uniform(*self.heading_diff_range))
            cpa = float(np.random.uniform(0, INTRUSION_DISTANCE * 1.5))
            tlosh = float(np.random.uniform(50, 1500))
            
            base_speed = np.random.uniform(*self.intruder_speed_range)
            performance = self.aircraft_performance
            speed = np.clip(base_speed, 
                          performance['min_speed'] * 0.8,
                          performance['max_speed'] * 1.2)
            
            bs.traf.creconfs(
                acid=f"INT{i:03d}", actype=self.aircraft_type, targetidx=target_idx,
                dpsi=dpsi, dcpa=cpa, tlosh=tlosh
            )
            
            idx = bs.traf.id2idx(f"INT{i:03d}")
            weather_factor = {"clear": 1.0, "turbulent": 0.9, "windy": 0.95}[self.weather_conditions]
            bs.traf.gs[idx] = speed * weather_factor

    def _get_obs(self):
        """Get observations with uncertainty measures"""
        idx = bs.traf.id2idx("OWN001")
        intr_ds, cos_b, sin_b = [], [], []
        x_sp, y_sp = [], []
        wpt_ds, wpt_qdr, cos_d, sin_d = [], [], [], []
        
        self.ac_hdg = bs.traf.hdg[idx]
        
        # Process intruders
        for i in range(self.num_intruders):
            intr_idx = i + 1
            if intr_idx < len(bs.traf.id):
                qdr, dis = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    bs.traf.lat[intr_idx], bs.traf.lon[intr_idx]
                )
                
                # Apply sensor noise and failures
                base_noise = np.random.normal(0, self.sensor_noise_std)
                if np.random.random() < self.sensor_failure_prob:
                    base_noise += np.random.normal(0, self.sensor_noise_std * 5)
                
                dist_km = dis * NM2KM + base_noise
                intr_ds.append(dist_km)
                
                b = fn.bound_angle_positive_negative_180(self.ac_hdg - qdr)
                cos_b.append(np.cos(np.deg2rad(b)))
                sin_b.append(np.sin(np.deg2rad(b)))
                
                hd_diff = self.ac_hdg - bs.traf.hdg[intr_idx]
                x_sp.append(-np.cos(np.deg2rad(hd_diff)) * bs.traf.gs[intr_idx])
                y_sp.append(bs.traf.gs[idx] - np.sin(np.deg2rad(hd_diff)) * bs.traf.gs[intr_idx])
                
                if dist_km < INTRUSION_DISTANCE:
                    self.true_conflict = True
            else:
                # Pad with safe values
                intr_ds.append(WAYPOINT_DISTANCE_MAX)
                cos_b.append(0.0)
                sin_b.append(0.0)
                x_sp.append(0.0)
                y_sp.append(0.0)
        
        # Process waypoints
        for lat, lon in zip(self.wpt_lat, self.wpt_lon):
            qdr, wpd = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx], lat, lon
            )
            wpt_qdr.append(qdr)
            wpt_ds.append(wpd * NM2KM)
            drift = fn.bound_angle_positive_negative_180(self.ac_hdg - qdr)
            cos_d.append(np.cos(np.deg2rad(drift)))
            sin_d.append(np.sin(np.deg2rad(drift)))
            self.average_drift = np.append(self.average_drift, abs(np.deg2rad(drift)))
        
        self.wpt_qdr = wpt_qdr
        self.waypoint_distance = wpt_ds
        
        # Build observation
        obs = {
            "intruder_distance": np.array(intr_ds[:self.num_intruders]) / WAYPOINT_DISTANCE_MAX,
            "cos_difference_pos": np.array(cos_b[:self.num_intruders]),
            "sin_difference_pos": np.array(sin_b[:self.num_intruders]),
            "x_difference_speed": np.array(x_sp[:self.num_intruders]) / AC_SPD,
            "y_difference_speed": np.array(y_sp[:self.num_intruders]) / AC_SPD,
            "waypoint_distance": np.array(wpt_ds) / WAYPOINT_DISTANCE_MAX,
            "cos_drift": np.array(cos_d),
            "sin_drift": np.array(sin_d),
        }
        
        # Add uncertainty metrics if enabled
        if self.uncertainty_injection:
            complexity = self._calculate_environment_complexity()
            violations, violation_score = self._detect_envelope_violation(obs)
            
            obs.update({
                "observation_uncertainty": np.array([self.sensor_noise_std + violation_score]),
                "environment_complexity": np.array([complexity]),
                "training_envelope_distance": np.array([violation_score]),
            })
            
            self.uncertainty_metrics = {
                'complexity': complexity,
                'violations': violations,
                'violation_score': violation_score,
                'sensor_noise': self.sensor_noise_std
            }
        
        return obs

    def _get_info(self):
        """Get info with hallucination detection metrics"""
        base_info = {
            "total_reward": self.total_reward,
            "total_intrusions": self.total_intrusions,
            "average_drift": self.average_drift.mean() if len(self.average_drift) > 0 else 0,
            "true_conflict": int(self.true_conflict),
            "aircraft_type": self.aircraft_type,
            "weather_conditions": self.weather_conditions,
            "timestep": self.timestep
        }
        
        if self.hallucination_detection:
            base_info.update(self.uncertainty_metrics)
            
            if len(self.action_history) > 5:
                action_variance = np.var(list(self.action_history))
                base_info["action_consistency"] = 1.0 / (1.0 + action_variance)
            
            min_distance = min(self.waypoint_distance) if self.waypoint_distance else WAYPOINT_DISTANCE_MAX
            base_info["safety_margin"] = min_distance / INTRUSION_DISTANCE
        
        return base_info

    def step(self, action):
        """Step with hallucination tracking and communication delay"""
        self.timestep += 1
        self.action_history.append(action[0])
        
        # Track envelope violations
        if self.track_training_envelope:
            obs = self._get_obs()
            violations, score = self._detect_envelope_violation(obs)
            if violations:
                self.envelope_violations.append({
                    'timestep': self.timestep,
                    'violations': violations,
                    'score': score
                })
        
        # Apply communication delay
        if self.communication_delay_range[1] > 0:
            delay_ms = np.random.randint(*self.communication_delay_range)
            delay_steps = max(1, delay_ms // 100)
            self._action_buffer.append((action[0], delay_steps))
            
            if self._action_buffer:
                cmd, remaining = self._action_buffer[0]
                if remaining > 1:
                    self._action_buffer[0] = (cmd, remaining-1)
                    use_action = 0  # No action this step
                else:
                    use_action = cmd
                    self._action_buffer.popleft()
            else:
                use_action = 0
        else:
            use_action = action[0]
        
        self._get_action(np.array([use_action]))
        
        # Advance simulation
        for _ in range(self.current_action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame()
        
        obs = self._get_obs()
        reward, done = self._get_reward()
        info = self._get_info()
        
        # Log episode data
        self.episode_data['timesteps'] = self.timestep
        self.episode_data['final_reward'] = self.total_reward
        
        if done:
            self.episode_data['violations'] = self.envelope_violations
            self.episode_data['total_intrusions'] = self.total_intrusions
            self.episode_data['min_safety_margin'] = info.get('safety_margin', 1.0)
            
            for acid in bs.traf.id:
                bs.traf.delete(bs.traf.id2idx(acid))
        
        return obs, reward, done, False, info

    def _get_reward(self):
        """Calculate reward with safety margin considerations"""
        reach_r = self._check_waypoint()
        drift_r = self._check_drift()
        intr_r = self._check_intrusion()
        
        envelope_penalty = 0
        if self.track_training_envelope and hasattr(self, 'uncertainty_metrics'):
            violation_score = self.uncertainty_metrics.get('violation_score', 0)
            if violation_score > self.envelope_violation_threshold:
                envelope_penalty = -0.5 * violation_score
        
        total = reach_r + drift_r + intr_r + envelope_penalty
        self.total_reward += total
        done = all(r == 1 for r in self.wpt_reach)
        
        return total, done

    def _generate_waypoint(self, acid="OWN001"):
        """Generate waypoint with stress testing parameters"""
        self.wpt_lat, self.wpt_lon, self.wpt_reach = [], [], []
        for _ in range(NUM_WAYPOINTS):
            wpt_dis = np.random.randint(
                int(WAYPOINT_DISTANCE_MIN * 0.8), 
                int(WAYPOINT_DISTANCE_MAX * 1.5)
            )
            idx = bs.traf.id2idx(acid)
            lat, lon = fn.get_point_at_distance(
                bs.traf.lat[idx], bs.traf.lon[idx], wpt_dis, 0
            )
            self.wpt_lat.append(lat)
            self.wpt_lon.append(lon)
            self.wpt_reach.append(0)

    def _check_waypoint(self):
        """Check waypoint reaching with aircraft-specific constraints"""
        r = 0
        performance = self.aircraft_performance
        size_factor = performance['length'] / 40.0
        adjusted_margin = DISTANCE_MARGIN * (1 + size_factor * 0.2)
        
        for i, dist in enumerate(self.waypoint_distance):
            if dist < adjusted_margin and not self.wpt_reach[i]:
                self.wpt_reach[i] = 1
                r += REACH_REWARD
        return r

    def _check_drift(self):
        """Check drift with aircraft-specific turn rates"""
        if len(self.average_drift) == 0:
            return 0
        
        performance = self.aircraft_performance
        maneuverability_factor = performance['turn_rate'] / 3.0
        adjusted_penalty = DRIFT_PENALTY * (2.0 - maneuverability_factor)
        
        return self.average_drift[-1] * adjusted_penalty

    def _check_intrusion(self):
        """Check intrusion with aircraft-specific separation"""
        r = 0
        idx = bs.traf.id2idx("OWN001")
        performance = self.aircraft_performance
        
        size_factor = performance['wing_span'] / 35.0
        adjusted_separation = INTRUSION_DISTANCE * (1 + size_factor * 0.1)
        
        for i in range(self.num_intruders):
            int_idx = i + 1
            if int_idx < len(bs.traf.id):
                _, d = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    bs.traf.lat[int_idx], bs.traf.lon[int_idx]
                )
                if d < adjusted_separation:
                    self.total_intrusions += 1
                    safety_violation_severity = (adjusted_separation - d) / adjusted_separation
                    penalty = INTRUSION_PENALTY * (1 + safety_violation_severity)
                    r += penalty
        return r

    def _get_action(self, action):
        """Process action with aircraft-specific constraints"""
        performance = self.aircraft_performance
        max_heading_change = D_HEADING * (performance['turn_rate'] / 3.0)
        constrained_action = np.clip(action[0], -1.0, 1.0)
        hdg_change = constrained_action * max_heading_change
        
        hdg = self.ac_hdg + hdg_change
        bs.stack.stack(f"HDG OWN001 {hdg}")

    def _render_frame(self):
        """Render frame with hallucination indicators"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption(f"ATC Sim - {self.aircraft_type} - {self.weather_conditions}")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        max_distance = 200
        canvas = pygame.Surface(self.window_size)
        
        bg_colors = {"clear": (135, 206, 235), "turbulent": (120, 150, 200), "windy": (150, 180, 220)}
        canvas.fill(bg_colors[self.weather_conditions])
        
        try:
            ac_idx = bs.traf.id2idx("OWN001")
            self._render_aircraft(canvas, max_distance, ac_idx)
            self._render_stress_indicators(canvas)
        except:
            pass
        
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def _render_aircraft(self, canvas, max_distance, ac_idx):
        """Render aircraft and intruders"""
        performance = self.aircraft_performance
        base_length = 8
        ac_len = base_length * (performance['length'] / 37.6)
        
        hdg_rad = np.deg2rad(bs.traf.hdg[ac_idx])
        dx = np.cos(hdg_rad) * ac_len / max_distance * self.window_width
        dy = np.sin(hdg_rad) * ac_len / max_distance * self.window_width
        
        colors = {"A320": (0, 0, 255), "B747": (255, 0, 0), "A380": (0, 255, 0), "E190": (255, 255, 0)}
        color = colors.get(self.aircraft_type, (0, 0, 0))
        
        pygame.draw.line(
            canvas, color,
            (self.window_width / 2 - dx / 2, self.window_height / 2 + dy / 2),
            (self.window_width / 2 + dx / 2, self.window_height / 2 - dy / 2),
            width=int(4 * (performance['mass'] / 78000))
        )
        
        # Render intruders
        for i in range(self.num_intruders):
            int_idx = i + 1
            if int_idx < len(bs.traf.id):
                int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                    bs.traf.lat[int_idx], bs.traf.lon[int_idx]
                )
                
                if int_dis < INTRUSION_DISTANCE * 0.5:
                    color = (255, 0, 0)
                elif int_dis < INTRUSION_DISTANCE:
                    color = (255, 165, 0)
                else:
                    color = (80, 80, 80)
                
                x_pos = (self.window_width/2) + (np.cos(np.deg2rad(int_qdr)) * (int_dis * NM2KM) / max_distance) * self.window_width
                y_pos = (self.window_height/2) - (np.sin(np.deg2rad(int_qdr)) * (int_dis * NM2KM) / max_distance) * self.window_height
                
                pygame.draw.circle(canvas, color, (int(x_pos), int(y_pos)), 5)
                pygame.draw.circle(
                    canvas, color,
                    (int(x_pos), int(y_pos)),
                    int((INTRUSION_DISTANCE * NM2KM / max_distance) * self.window_width),
                    width=1
                )
        
        # Render waypoints
        for qdr, dis, reach in zip(self.wpt_qdr, self.waypoint_distance, self.wpt_reach):
            circle_x = ((np.cos(np.deg2rad(qdr)) * dis) / max_distance) * self.window_width
            circle_y = ((np.sin(np.deg2rad(qdr)) * dis) / max_distance) * self.window_width
            
            color = (100, 255, 100) if reach else (255, 255, 255)
            
            pygame.draw.circle(
                canvas, color,
                (int(self.window_width/2 + circle_x), int(self.window_height/2 - circle_y)),
                radius=6, width=0
            )

    def _render_stress_indicators(self, canvas):
        """Render stress testing and hallucination indicators"""
        font = pygame.font.Font(None, 24)
        
        if hasattr(self, 'uncertainty_metrics'):
            complexity = self.uncertainty_metrics.get('complexity', 0)
            violation_score = self.uncertainty_metrics.get('violation_score', 0)
            
            complexity_text = font.render(f"Complexity: {complexity:.2f}", True, (255, 255, 255))
            canvas.blit(complexity_text, (10, 10))
            
            if violation_score > self.envelope_violation_threshold:
                violation_text = font.render("ENVELOPE VIOLATION", True, (255, 0, 0))
                canvas.blit(violation_text, (10, 40))
        
        type_text = font.render(f"Aircraft: {self.aircraft_type}", True, (255, 255, 255))
        canvas.blit(type_text, (10, self.window_height - 30))

    def close(self):
        """Enhanced cleanup with episode data logging"""
        if hasattr(self, 'episode_data'):
            self.episode_data['end_time'] = time.time()
            self.episode_data['duration'] = self.episode_data['end_time'] - self.episode_data['start_time']
            
            # Save episode data to JSON
            os.makedirs('logs', exist_ok=True)
            fname = f"logs/episode_{int(self.episode_data['start_time'])}.json"
            with open(fname, "w") as f:
                json.dump(self.episode_data, f, indent=2)
        
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
