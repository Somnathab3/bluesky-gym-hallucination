# bluesky_gym/envs/custom_vertical_cr_env.py
"""
Custom Vertical Conflict Resolution Environment for ML Hallucination Research

This environment is designed for researching ML-based hallucination effects 
on safety margins in air traffic control, specifically for vertical 
conflict resolution scenarios during descent operations.

Key features:
- Enhanced observation space for hallucination detection
- Configurable vertical safety margins for stress testing
- Detailed logging for ML model behavior analysis
- Support for boundary condition testing in vertical domain
"""

import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

# Environment constants
DISTANCE_MARGIN = 5  # km
NM2KM = 1.852

INTRUSION_PENALTY = -50
ALT_DIF_REWARD_SCALE = -5/3000
CRASH_PENALTY = -100
RWY_ALT_DIF_REWARD_SCALE = -50/3000
HALLUCINATION_PENALTY = -1  # Additional penalty for potential hallucinations

NUM_INTRUDERS = 5
INTRUSION_DISTANCE = 5  # NM
VERTICAL_MARGIN = 1000 * 0.3048  # ft converted to meters

# Define constants for aircraft parameters
ALT_MEAN = 1500
ALT_STD = 3000
VZ_MEAN = 0
VZ_STD = 5
RWY_DIS_MEAN = 100
RWY_DIS_STD = 200
DEFAULT_RWY_DIS = 200 
RWY_LAT = 52
RWY_LON = 4

ACTION_2_MS = 12.5  # approx 2500 ft/min
ALT_MIN = 2000
ALT_MAX = 4000
TARGET_ALT_DIF = 500
AC_SPD = 150
ACTION_FREQUENCY = 30

# Vertical safety margin configurations for testing
VERTICAL_SAFETY_MARGINS = {
    'conservative': 1500 * 0.3048,  # 1500 ft
    'standard': 1000 * 0.3048,      # 1000 ft  
    'aggressive': 500 * 0.3048,     # 500 ft
    'critical': 200 * 0.3048        # 200 ft for stress testing
}

class CustomVerticalCREnv(gym.Env):
    """
    Enhanced Vertical Conflict Resolution Environment for ML Hallucination Research
    
    This environment builds upon the original VerticalCREnv with additional features
    for studying ML hallucination effects and safety margin analysis in the vertical domain.
    
    Key additions:
    - Configurable vertical safety margins for stress testing
    - Enhanced observation space with confidence metrics
    - Hallucination detection and logging
    - Boundary condition testing capabilities for descent scenarios
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 120}

    def __init__(self, render_mode=None, vertical_safety_margin='standard', 
                 enable_hallucination_detection=True, 
                 boundary_test_mode=False,
                 stress_test_mode=False):
        """
        Initialize the Custom Vertical CR Environment
        
        Args:
            render_mode: Rendering mode ('rgb_array', 'human', or None)
            vertical_safety_margin: Vertical safety margin level ('conservative', 'standard', 'aggressive', 'critical')
            enable_hallucination_detection: Enable hallucination detection features
            boundary_test_mode: Enable boundary condition testing
            stress_test_mode: Enable stress testing with extreme scenarios
        """
        self.window_width = 512
        self.window_height = 256
        self.window_size = (self.window_width, self.window_height)
        
        # Configuration parameters
        self.vertical_safety_margin_level = vertical_safety_margin
        self.vertical_margin = VERTICAL_SAFETY_MARGINS[vertical_safety_margin]
        self.enable_hallucination_detection = enable_hallucination_detection
        self.boundary_test_mode = boundary_test_mode
        self.stress_test_mode = stress_test_mode
        
        # Enhanced observation space for hallucination research
        obs_dict = {
            # Runway information
            "altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
            "vz": spaces.Box(-np.inf, np.inf, dtype=np.float64),
            "target_altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
            "runway_distance": spaces.Box(-np.inf, np.inf, dtype=np.float64),
            # Intruder information
            "intruder_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "altitude_difference": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "x_difference_speed": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "y_difference_speed": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "z_difference_speed": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64)
        }
        
        # Add hallucination detection features if enabled
        if self.enable_hallucination_detection:
            obs_dict.update({
                "observation_confidence": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "boundary_proximity": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "anomaly_score": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "vertical_safety_ratio": spaces.Box(0.0, 10.0, shape=(1,), dtype=np.float64),
                "descent_rate_anomaly": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "altitude_boundary_risk": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
            })
        
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize BlueSky
        bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')

        # Logging and analysis variables
        self.total_reward = 0
        self.total_intrusions = 0
        self.final_altitude = 0
        self.hallucination_events = []
        self.boundary_violations = 0
        self.vertical_safety_violations = 0
        self.extreme_descent_events = 0
        
        # Observation history for hallucination detection
        self.observation_history = []
        self.max_history_length = 10
        
        # Target altitude for current episode
        self.target_alt = None
        
        # Rendering
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        bs.traf.reset()

        # Reset logging variables
        self.total_reward = 0
        self.total_intrusions = 0
        self.final_altitude = 0
        self.hallucination_events = []
        self.boundary_violations = 0
        self.vertical_safety_violations = 0
        self.extreme_descent_events = 0
        self.observation_history = []

        # Set initial and target altitudes
        if self.stress_test_mode:
            # More challenging altitude scenarios for stress testing
            alt_init = np.random.randint(ALT_MAX - 500, ALT_MAX)
            self.target_alt = np.random.randint(ALT_MIN, alt_init - 1000)
        else:
            alt_init = np.random.randint(ALT_MIN, ALT_MAX)
            self.target_alt = alt_init + np.random.randint(-TARGET_ALT_DIF, TARGET_ALT_DIF)

        # Create ownship
        bs.traf.cre('KL001', actype="A320", acalt=alt_init, acspd=AC_SPD)
        bs.traf.swvnav[0] = False

        # Generate conflicts
        self._generate_conflicts(acid='KL001')

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self._get_action(action)

        # Execute multiple simulation steps per action
        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame()
                observation = self._get_obs()

        observation = self._get_obs()
        reward, terminated = self._get_reward()
        
        # Detect potential hallucinations
        if self.enable_hallucination_detection:
            self._detect_hallucinations(observation, action)

        info = self._get_info()

        # Clean up if terminated
        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, False, info

    def _generate_conflicts(self, acid='KL001'):
        """Generate conflicting aircraft with enhanced boundary testing for vertical scenarios"""
        target_idx = bs.traf.id2idx(acid)
        altitude = bs.traf.alt[target_idx]
        spd = bs.traf.gs[target_idx]
        
        for i in range(NUM_INTRUDERS):
            if self.boundary_test_mode:
                # Generate more challenging boundary scenarios
                dpsi = np.random.choice([45, 90, 135, 180, 225, 270, 315])
                cpa = np.random.uniform(0.5, INTRUSION_DISTANCE * 1.2)
                tlosh = np.random.randint(50, 150)  # Shorter time to conflict
            else:
                dpsi = np.random.randint(45, 315)
                cpa = np.random.randint(0, INTRUSION_DISTANCE)
                tlosh = np.random.randint(100, int((DEFAULT_RWY_DIS * 0.9) * 1000 / spd))
            
            # Enhanced altitude conflict generation
            average_tod = (DEFAULT_RWY_DIS * 1000 / spd) - 2 * self.target_alt / ACTION_2_MS
            
            if self.stress_test_mode:
                # Create more challenging vertical conflicts
                if tlosh > average_tod:
                    dH = np.random.randint(int(-altitude + 200), int((self.target_alt - altitude) + 50))
                else:
                    dH = np.random.randint(int((self.target_alt - altitude) - 200), int((self.target_alt - altitude) + 200))
            else:
                if tlosh > average_tod:
                    dH = np.random.randint(int(-altitude + 500), int((self.target_alt - altitude) + 100))
                else:
                    dH = np.random.randint(int((self.target_alt - altitude) - 500), int((self.target_alt - altitude) + 500))
            
            tlosv = 100000000000.

            bs.traf.creconfs(acid=f'{i}', actype="A320", targetidx=target_idx,
                           dpsi=dpsi, dcpa=cpa, tlosh=tlosh, dH=dH, tlosv=tlosv)
            bs.traf.alt[i+1] = bs.traf.alt[target_idx] + dH
            bs.traf.ap.selaltcmd(i+1, bs.traf.alt[target_idx] + dH, 0)

    def _get_obs(self):
        """Get enhanced observations including hallucination detection features"""
        ac_idx = bs.traf.id2idx('KL001')

        # Standard observations
        self.intruder_distance = []
        self.cos_bearing = []
        self.sin_bearing = []
        self.altitude_difference = []
        self.x_difference_speed = []
        self.y_difference_speed = []
        self.z_difference_speed = []

        self.ac_hdg = bs.traf.hdg[ac_idx]
        self.altitude = bs.traf.alt[0]
        self.vz = bs.traf.vs[0]

        # Process intruder information
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )

            self.intruder_distance.append(int_dis * NM2KM)

            alt_dif = bs.traf.alt[int_idx] - self.altitude
            vz_dif = bs.traf.vs[int_idx] - self.vz

            self.altitude_difference.append(alt_dif)
            self.z_difference_speed.append(vz_dif)

            bearing = self.ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.sin_bearing.append(np.sin(np.deg2rad(bearing)))

            heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
            x_dif = -np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

            self.x_difference_speed.append(x_dif)
            self.y_difference_speed.append(y_dif)
        
        # Calculate runway distance
        self.runway_distance = (DEFAULT_RWY_DIS - bs.tools.geo.kwikdist(
            RWY_LAT, RWY_LON, bs.traf.lat[0], bs.traf.lon[0]) * NM2KM)

        # Normalized observations
        obs_altitude = np.array([(self.altitude - ALT_MEAN) / ALT_STD])
        obs_vz = np.array([(self.vz - VZ_MEAN) / VZ_STD])
        obs_target_alt = np.array([((self.target_alt - ALT_MEAN) / ALT_STD)])
        obs_runway_distance = np.array([(self.runway_distance - RWY_DIS_MEAN) / RWY_DIS_STD])

        # Base observation
        observation = {
            "altitude": obs_altitude,
            "vz": obs_vz,
            "target_altitude": obs_target_alt,
            "runway_distance": obs_runway_distance,
            "intruder_distance": np.array(self.intruder_distance) / DEFAULT_RWY_DIS,
            "cos_difference_pos": np.array(self.cos_bearing),
            "sin_difference_pos": np.array(self.sin_bearing),
            "altitude_difference": np.array(self.altitude_difference) / ALT_STD,
            "x_difference_speed": np.array(self.x_difference_speed) / AC_SPD,
            "y_difference_speed": np.array(self.y_difference_speed) / AC_SPD,
            "z_difference_speed": np.array(self.z_difference_speed)
        }
        
        # Add hallucination detection features
        if self.enable_hallucination_detection:
            observation.update(self._get_hallucination_features(observation))
        
        # Store observation history
        self.observation_history.append(observation.copy())
        if len(self.observation_history) > self.max_history_length:
            self.observation_history.pop(0)
        
        return observation

    def _get_hallucination_features(self, base_obs):
        """Calculate hallucination detection features for vertical domain"""
        # Calculate observation confidence based on consistency
        confidence = self._calculate_observation_confidence()
        
        # Calculate boundary proximity for vertical scenarios
        boundary_proximity = self._calculate_boundary_proximity(base_obs)
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(base_obs)
        
        # Calculate vertical safety margin ratio
        min_vertical_separation = float('inf')
        for i, alt_diff in enumerate(self.altitude_difference):
            horizontal_dist = self.intruder_distance[i] * DEFAULT_RWY_DIS
            if horizontal_dist < INTRUSION_DISTANCE:
                min_vertical_separation = min(min_vertical_separation, abs(alt_diff))
        
        vertical_safety_ratio = (min_vertical_separation / self.vertical_margin 
                               if min_vertical_separation != float('inf') else 5.0)
        
        # Detect extreme descent rate anomalies
        descent_rate_anomaly = self._calculate_descent_rate_anomaly()
        
        # Calculate altitude boundary risk
        altitude_boundary_risk = self._calculate_altitude_boundary_risk()
        
        return {
            "observation_confidence": np.array([confidence]),
            "boundary_proximity": np.array([boundary_proximity]),
            "anomaly_score": np.array([anomaly_score]),
            "vertical_safety_ratio": np.array([min(10.0, vertical_safety_ratio)]),
            "descent_rate_anomaly": np.array([descent_rate_anomaly]),
            "altitude_boundary_risk": np.array([altitude_boundary_risk])
        }

    def _calculate_observation_confidence(self):
        """Calculate confidence in observations based on historical consistency"""
        if len(self.observation_history) < 2:
            return 1.0
        
        # Check consistency in altitude and vertical speed
        current_alt = self.altitude
        current_vz = self.vz
        
        if len(self.observation_history) > 0:
            prev_alt = self.observation_history[-1]["altitude"][0] * ALT_STD + ALT_MEAN
            prev_vz = self.observation_history[-1]["vz"][0] * VZ_STD + VZ_MEAN
            
            # Check for unrealistic changes
            alt_change = abs(current_alt - prev_alt)
            vz_change = abs(current_vz - prev_vz)
            
            # Penalize sudden unrealistic changes
            alt_inconsistency = min(1.0, alt_change / 100.0)  # 100m threshold
            vz_inconsistency = min(1.0, vz_change / 10.0)     # 10 m/s threshold
            
            confidence = max(0.0, 1.0 - (alt_inconsistency + vz_inconsistency) / 2.0)
            return confidence
        
        return 1.0

    def _calculate_boundary_proximity(self, obs):
        """Calculate how close observations are to training data boundaries for vertical domain"""
        # Check altitude boundaries
        normalized_alt = obs["altitude"][0]
        alt_boundary = max(0.0, abs(normalized_alt) - 1.5)  # Beyond 1.5 std deviations
        
        # Check vertical speed boundaries
        normalized_vz = obs["vz"][0]
        vz_boundary = max(0.0, abs(normalized_vz) - 2.0)    # Beyond 2 std deviations
        
        # Check runway distance boundaries
        normalized_rwy_dist = obs["runway_distance"][0]
        rwy_boundary = max(0.0, abs(normalized_rwy_dist) - 2.0)
        
        boundary_proximity = min(1.0, (alt_boundary + vz_boundary + rwy_boundary) / 3.0)
        return boundary_proximity

    def _calculate_anomaly_score(self, obs):
        """Calculate anomaly score for potential hallucination detection in vertical domain"""
        if len(self.observation_history) < 3:
            return 0.0
        
        # Check for sudden jumps in altitude differences
        current_alt_diffs = obs["altitude_difference"]
        historical_alt_diffs = [h["altitude_difference"] for h in self.observation_history[-3:]]
        
        if len(historical_alt_diffs) > 0:
            historical_mean = np.mean(historical_alt_diffs, axis=0)
            historical_std = np.std(historical_alt_diffs, axis=0) + 1e-6
            
            z_scores = np.abs((current_alt_diffs - historical_mean) / historical_std)
            anomaly_score = np.mean(z_scores > 2.5)  # Values > 2.5 std deviations
            
            return min(1.0, anomaly_score)
        
        return 0.0

    def _calculate_descent_rate_anomaly(self):
        """Detect anomalous descent rates that might indicate hallucinations"""
        # Check for extreme descent rates
        max_safe_descent_rate = 15.0  # m/s (approximately 3000 fpm)
        
        if abs(self.vz) > max_safe_descent_rate:
            self.extreme_descent_events += 1
            return 1.0
        
        # Check for rapid changes in descent rate
        if len(self.observation_history) > 0:
            prev_vz = self.observation_history[-1]["vz"][0] * VZ_STD + VZ_MEAN
            vz_change_rate = abs(self.vz - prev_vz)
            
            if vz_change_rate > 5.0:  # Rapid change threshold
                return min(1.0, vz_change_rate / 10.0)
        
        return 0.0

    def _calculate_altitude_boundary_risk(self):
        """Calculate risk of altitude boundary violations"""
        # Risk of ground collision
        ground_risk = max(0.0, (500.0 - self.altitude) / 500.0) if self.altitude < 500.0 else 0.0
        
        # Risk of extreme altitude
        ceiling_risk = max(0.0, (self.altitude - 6000.0) / 1000.0) if self.altitude > 6000.0 else 0.0
        
        return min(1.0, ground_risk + ceiling_risk)

    def _detect_hallucinations(self, observation, action):
        """Detect potential ML hallucinations in vertical domain"""
        if not self.enable_hallucination_detection:
            return
        
        hallucination_detected = False
        
        # 1. Inconsistent observations
        if observation.get("observation_confidence", [1.0])[0] < 0.3:
            hallucination_detected = True
            
        # 2. Boundary violations
        if observation.get("boundary_proximity", [0.0])[0] > 0.7:
            self.boundary_violations += 1
            hallucination_detected = True
            
        # 3. High anomaly scores
        if observation.get("anomaly_score", [0.0])[0] > 0.8:
            hallucination_detected = True
        
        # 4. Vertical safety margin violations
        vertical_safety_ratio = observation.get("vertical_safety_ratio", [1.0])[0]
        if vertical_safety_ratio < 0.3:
            self.vertical_safety_violations += 1
            hallucination_detected = True
        
        # 5. Descent rate anomalies
        if observation.get("descent_rate_anomaly", [0.0])[0] > 0.5:
            hallucination_detected = True
            
        # 6. Altitude boundary risks
        if observation.get("altitude_boundary_risk", [0.0])[0] > 0.8:
            hallucination_detected = True
        
        if hallucination_detected:
            self.hallucination_events.append({
                'step': len(self.observation_history),
                'confidence': observation.get("observation_confidence", [0.0])[0],
                'boundary_proximity': observation.get("boundary_proximity", [0.0])[0],
                'anomaly_score': observation.get("anomaly_score", [0.0])[0],
                'vertical_safety_ratio': vertical_safety_ratio,
                'descent_rate_anomaly': observation.get("descent_rate_anomaly", [0.0])[0],
                'altitude_boundary_risk': observation.get("altitude_boundary_risk", [0.0])[0],
                'action': action[0] if hasattr(action, '__len__') else action,
                'altitude': self.altitude,
                'vertical_speed': self.vz
            })

    def _get_info(self):
        """Get environment info including hallucination metrics"""
        base_info = {
            "total_reward": self.total_reward,
            "total_intrusions": self.total_intrusions,
            "final_altitude": self.final_altitude,
            "vertical_safety_margin_level": self.vertical_safety_margin_level,
            "vertical_margin": self.vertical_margin,
            "target_altitude": self.target_alt,
            "current_altitude": self.altitude,
            "vertical_speed": self.vz
        }
        
        if self.enable_hallucination_detection:
            base_info.update({
                'hallucination_events': len(self.hallucination_events),
                'boundary_violations': self.boundary_violations,
                'vertical_safety_violations': self.vertical_safety_violations,
                'extreme_descent_events': self.extreme_descent_events,
                'hallucination_rate': len(self.hallucination_events) / max(1, len(self.observation_history))
            })
        
        return base_info
    
    def _get_reward(self):
        """Calculate reward with hallucination penalties"""
        int_penalty = self._check_intrusion()
        done = 0
        
        if self.runway_distance > 0 and self.altitude > 0:
            alt_penalty = abs(self.target_alt - self.altitude) * ALT_DIF_REWARD_SCALE
        elif self.altitude <= 0:
            alt_penalty = CRASH_PENALTY
            self.final_altitude = -100
            done = 1
        elif self.runway_distance <= 0:
            alt_penalty = self.altitude * RWY_ALT_DIF_REWARD_SCALE
            self.final_altitude = self.altitude
            done = 1
        else:
            alt_penalty = 0
            
        # Add hallucination penalty
        hallucination_reward = 0
        if self.enable_hallucination_detection and len(self.hallucination_events) > 0:
            recent_hallucinations = sum(1 for event in self.hallucination_events 
                                      if event['step'] >= len(self.observation_history) - 1)
            hallucination_reward = recent_hallucinations * HALLUCINATION_PENALTY
            
        reward = alt_penalty + int_penalty + hallucination_reward
        self.total_reward += reward
        return reward, done

    def _check_intrusion(self):
        """Check for intrusion violations using configured vertical safety margin"""
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            _, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )
            vert_dis = bs.traf.alt[ac_idx] - bs.traf.alt[int_idx]
            
            if int_dis < INTRUSION_DISTANCE and abs(vert_dis) < self.vertical_margin:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        
        return reward
        
    def _get_action(self, action):
        """Convert action to BlueSky vertical speed command"""
        # Transform action to meters per second
        action_value = action[0] if hasattr(action, '__len__') else action
        action_ms = action_value * ACTION_2_MS

        # BlueSky interprets vertical velocity command through altitude commands 
        # with a vertical speed (magnitude). Check sign of action and give arbitrary 
        # altitude command
        if action_ms >= 0:
            bs.traf.selalt[0] = 1000000  # High target altitude to start climb
            bs.traf.selvs[0] = action_ms
        elif action_ms < 0:
            bs.traf.selalt[0] = 0  # Low target altitude to start descent
            bs.traf.selvs[0] = action_ms

    def _render_frame(self):
        """Render the environment visualization with hallucination indicators"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        zero_offset = 25
        max_distance = 180  # width of screen in km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135, 206, 235))

        # Draw ground surface
        pygame.draw.rect(canvas, (154, 205, 50),
            pygame.Rect((0, self.window_height - 50), (self.window_width, 50))
        )
        
        # Draw target altitude line
        max_alt = 5000
        target_alt_y = int((-1 * (self.target_alt - max_alt) / max_alt) * (self.window_height - 50))

        pygame.draw.line(canvas, (255, 255, 255),
            (0, target_alt_y), (self.window_width, target_alt_y)
        )

        # Draw runway
        runway_length = 30
        runway_start = int(((self.runway_distance + zero_offset) / max_distance) * self.window_width)
        runway_end = int(runway_start + (runway_length / max_distance) * self.window_width)

        pygame.draw.line(canvas, (119, 136, 153),
            (runway_start, self.window_height - 50),
            (runway_end, self.window_height - 50),
            width=3
        )

        # Draw ownship aircraft
        aircraft_alt = int((-1 * (self.altitude - max_alt) / max_alt) * (self.window_height - 50))
        aircraft_start = int(((zero_offset) / max_distance) * self.window_width)
        aircraft_end = int(aircraft_start + (4 / max_distance) * self.window_width)

        # Color ownship based on hallucination risk
        ownship_color = (0, 0, 0)  # Default black
        if self.enable_hallucination_detection and len(self.hallucination_events) > 0:
            recent_events = [e for e in self.hallucination_events if e['step'] >= len(self.observation_history) - 3]
            if recent_events:
                ownship_color = (255, 165, 0)  # Orange for hallucination warning

        pygame.draw.line(canvas, ownship_color,
            (aircraft_start, aircraft_alt), (aircraft_end, aircraft_alt),
            width=5
        )

        # Draw intruders with enhanced safety margin visualization
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            int_alt = int((-1 * (bs.traf.alt[int_idx] - max_alt) / max_alt) * (self.window_height - 50))
            int_x_dis = self.intruder_distance[int_idx - 1] * self.cos_bearing[int_idx - 1]
            int_y_dis = self.intruder_distance[int_idx - 1] * self.sin_bearing[int_idx - 1]
            
            width_temp = int(5 + int_y_dis / 20)
            aircraft_start = int(((zero_offset + int_x_dis) / max_distance) * self.window_width)
            aircraft_end = int(aircraft_start + (4 / max_distance) * self.window_width)
            
            # Enhanced color coding based on safety margins
            vertical_separation = abs(bs.traf.alt[0] - bs.traf.alt[int_idx])
            horizontal_separation = abs(int_y_dis)
            
            if (horizontal_separation < DISTANCE_MARGIN and 
                vertical_separation < self.vertical_margin):
                color = (255, 0, 0)      # Red - critical conflict
            elif (horizontal_separation < INTRUSION_DISTANCE and 
                  vertical_separation < self.vertical_margin * 1.5):
                color = (255, 165, 0)    # Orange - warning
            elif vertical_separation < self.vertical_margin * 2:
                color = (255, 255, 0)    # Yellow - caution
            else:
                color = (255, 255, 255)  # White - safe

            pygame.draw.line(canvas, color,
                (aircraft_start, int_alt), (aircraft_end, int_alt),
                width=width_temp
            )

            # Draw vertical safety margin boxes
            hor_margin = (DISTANCE_MARGIN * NM2KM / max_distance) * self.window_width
            ver_margin = (self.vertical_margin / max_alt) * self.window_height

            # Draw safety margin rectangle
            pygame.draw.line(canvas, 'black',
                (aircraft_start - hor_margin/2, int_alt - ver_margin),
                (aircraft_end + hor_margin/2, int_alt - ver_margin),
                width=1
            )
            pygame.draw.line(canvas, 'black',
                (aircraft_start - hor_margin/2, int_alt + ver_margin),
                (aircraft_end + hor_margin/2, int_alt + ver_margin),
                width=1
            )
            pygame.draw.line(canvas, 'black',
                (aircraft_start - hor_margin/2, int_alt - ver_margin),
                (aircraft_start - hor_margin/2, int_alt + ver_margin),
                width=1
            )
            pygame.draw.line(canvas, 'black',
                (aircraft_end + hor_margin/2, int_alt - ver_margin),
                (aircraft_end + hor_margin/2, int_alt + ver_margin),
                width=1
            )

        # Draw hallucination and safety indicators
        if self.enable_hallucination_detection:
            # Hallucination indicator
            if len(self.hallucination_events) > 0:
                recent_events = [e for e in self.hallucination_events if e['step'] >= len(self.observation_history) - 5]
                if recent_events:
                    pygame.draw.circle(canvas, (255, 255, 0),
                        (50, 50), radius=20, width=3
                    )
                    # Add text "HALL" in the circle
                    font = pygame.font.Font(None, 24)
                    text = font.render("HALL", True, (255, 255, 0))
                    canvas.blit(text, (35, 42))
            
            # Safety margin indicator
            if self.vertical_safety_violations > 0:
                pygame.draw.circle(canvas, (255, 0, 0),
                    (100, 50), radius=20, width=3
                )
                font = pygame.font.Font(None, 24)
                text = font.render("SAFE", True, (255, 0, 0))
                canvas.blit(text, (85, 42))
            
            # Boundary violation indicator
            if self.boundary_violations > 0:
                pygame.draw.circle(canvas, (255, 165, 0),
                    (150, 50), radius=20, width=3
                )
                font = pygame.font.Font(None, 24)
                text = font.render("BOUN", True, (255, 165, 0))
                canvas.blit(text, (135, 42))

        # Display current statistics
        if self.enable_hallucination_detection:
            font = pygame.font.Font(None, 24)
            
            # Hallucination count
            hall_text = font.render(f"Hallucinations: {len(self.hallucination_events)}", True, (255, 255, 255))
            canvas.blit(hall_text, (10, self.window_height - 40))
            
            # Safety violations
            safe_text = font.render(f"Safety Violations: {self.vertical_safety_violations}", True, (255, 255, 255))
            canvas.blit(safe_text, (10, self.window_height - 20))
            
            # Current altitude and vertical speed
            alt_text = font.render(f"Alt: {int(self.altitude)}m VS: {self.vz:.1f}m/s", True, (255, 255, 255))
            canvas.blit(alt_text, (250, self.window_height - 40))
            
            # Safety margin level
            margin_text = font.render(f"Safety: {self.vertical_safety_margin_level}", True, (255, 255, 255))
            canvas.blit(margin_text, (250, self.window_height - 20))

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        """Close the environment"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
