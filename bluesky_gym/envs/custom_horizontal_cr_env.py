# bluesky_gym/envs/custom_horizontal_cr_env.py
"""
Custom Horizontal Conflict Resolution Environment for ML Hallucination Research

This environment is designed for researching ML-based hallucination effects 
on safety margins in air traffic control, specifically for horizontal 
conflict resolution scenarios.

Key features:
- Enhanced observation space for hallucination detection
- Configurable safety margins for stress testing
- Detailed logging for ML model behavior analysis
- Support for boundary condition testing
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
REACH_REWARD = 1
DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1
HALLUCINATION_PENALTY = -0.2  # Additional penalty for potential hallucinations

NUM_INTRUDERS = 5
NUM_WAYPOINTS = 1
INTRUSION_DISTANCE = 5  # NM

WAYPOINT_DISTANCE_MIN = 100
WAYPOINT_DISTANCE_MAX = 150

D_HEADING = 45
AC_SPD = 150
NM2KM = 1.852
ACTION_FREQUENCY = 10

# Safety margin configurations for testing
SAFETY_MARGIN_LEVELS = {
    'conservative': 8.0,  # NM
    'standard': 5.0,      # NM  
    'aggressive': 3.0,    # NM
    'critical': 1.5       # NM for stress testing
}

class CustomHorizontalCREnv(gym.Env):
    """
    Enhanced Horizontal Conflict Resolution Environment for ML Hallucination Research
    
    This environment builds upon the original HorizontalCREnv with additional features
    for studying ML hallucination effects and safety margin analysis.
    
    Key additions:
    - Configurable safety margins for stress testing
    - Enhanced observation space with confidence metrics
    - Hallucination detection and logging
    - Boundary condition testing capabilities
    """
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 120}

    def __init__(self, render_mode=None, safety_margin='standard', 
                 enable_hallucination_detection=True, 
                 boundary_test_mode=False):
        """
        Initialize the Custom Horizontal CR Environment
        
        Args:
            render_mode: Rendering mode ('rgb_array', 'human', or None)
            safety_margin: Safety margin level ('conservative', 'standard', 'aggressive', 'critical')
            enable_hallucination_detection: Enable hallucination detection features
            boundary_test_mode: Enable boundary condition testing
        """
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)
        
        # Configuration parameters
        self.safety_margin_level = safety_margin
        self.intrusion_distance = SAFETY_MARGIN_LEVELS[safety_margin]
        self.enable_hallucination_detection = enable_hallucination_detection
        self.boundary_test_mode = boundary_test_mode
        
        # Enhanced observation space for hallucination research
        obs_dict = {
            "intruder_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "x_difference_speed": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "y_difference_speed": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "waypoint_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "cos_drift": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "sin_drift": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
        }
        
        # Add hallucination detection features if enabled
        if self.enable_hallucination_detection:
            obs_dict.update({
                "observation_confidence": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "boundary_proximity": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "anomaly_score": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "safety_margin_ratio": spaces.Box(0.0, 5.0, shape=(1,), dtype=np.float64),
            })
        
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize BlueSky
        bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 5;FF')

        # Logging and analysis variables
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])
        self.hallucination_events = []
        self.boundary_violations = 0
        self.safety_margin_violations = 0
        
        # Observation history for hallucination detection
        self.observation_history = []
        self.max_history_length = 10
        
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
        self.average_drift = np.array([])
        self.hallucination_events = []
        self.boundary_violations = 0
        self.safety_margin_violations = 0
        self.observation_history = []

        # Create ownship
        bs.traf.cre('KL001', actype="A320", acspd=AC_SPD)

        # Generate conflicts and waypoints
        self._generate_conflicts()
        self._generate_waypoint()
        
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
                observation = self._get_obs()
                self._render_frame()

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
        """Generate conflicting aircraft with enhanced boundary testing"""
        target_idx = bs.traf.id2idx(acid)
        
        for i in range(NUM_INTRUDERS):
            if self.boundary_test_mode:
                # Generate more challenging boundary scenarios
                dpsi = np.random.choice([45, 90, 135, 180, 225, 270, 315])
                cpa = np.random.uniform(0.5, self.intrusion_distance * 1.2)
                tlosh = np.random.randint(80, 200)  # Shorter time to conflict
            else:
                dpsi = np.random.randint(45, 315)
                cpa = np.random.randint(0, int(self.intrusion_distance))
                tlosh = np.random.randint(100, 1000)
            
            bs.traf.creconfs(acid=f'{i}', actype="A320", targetidx=target_idx, 
                           dpsi=dpsi, dcpa=cpa, tlosh=tlosh)

    def _generate_waypoint(self, acid='KL001'):
        """Generate waypoints for navigation"""
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []
        
        for i in range(NUM_WAYPOINTS):
            wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
            wpt_hdg_init = 0

            ac_idx = bs.traf.id2idx(acid)
            wpt_lat, wpt_lon = fn.get_point_at_distance(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init
            )
            
            self.wpt_lat.append(wpt_lat)
            self.wpt_lon.append(wpt_lon)
            self.wpt_reach.append(0)

    def _get_obs(self):
        """Get enhanced observations including hallucination detection features"""
        ac_idx = bs.traf.id2idx('KL001')

        # Standard observations
        self.intruder_distance = []
        self.cos_bearing = []
        self.sin_bearing = []
        self.x_difference_speed = []
        self.y_difference_speed = []

        self.waypoint_distance = []
        self.wpt_qdr = []
        self.cos_drift = []
        self.sin_drift = []
        self.drift = []

        self.ac_hdg = bs.traf.hdg[ac_idx]

        # Process intruder information
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )
        
            self.intruder_distance.append(int_dis * NM2KM)

            bearing = self.ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.sin_bearing.append(np.sin(np.deg2rad(bearing)))

            heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
            x_dif = -np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

            self.x_difference_speed.append(x_dif)
            self.y_difference_speed.append(y_dif)

        # Process waypoint information
        for lat, lon in zip(self.wpt_lat, self.wpt_lon):
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon
            )
        
            self.waypoint_distance.append(wpt_dis * NM2KM)
            self.wpt_qdr.append(wpt_qdr)

            drift = self.ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)

            self.drift.append(drift)
            self.cos_drift.append(np.cos(np.deg2rad(drift)))
            self.sin_drift.append(np.sin(np.deg2rad(drift)))

        # Base observation
        observation = {
            "intruder_distance": np.array(self.intruder_distance) / WAYPOINT_DISTANCE_MAX,
            "cos_difference_pos": np.array(self.cos_bearing),
            "sin_difference_pos": np.array(self.sin_bearing),
            "x_difference_speed": np.array(self.x_difference_speed) / AC_SPD,
            "y_difference_speed": np.array(self.y_difference_speed) / AC_SPD,
            "waypoint_distance": np.array(self.waypoint_distance) / WAYPOINT_DISTANCE_MAX,
            "cos_drift": np.array(self.cos_drift),
            "sin_drift": np.array(self.sin_drift)
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
        """Calculate hallucination detection features"""
        # Calculate observation confidence based on consistency
        confidence = self._calculate_observation_confidence()
        
        # Calculate boundary proximity (how close to training data boundaries)
        boundary_proximity = self._calculate_boundary_proximity(base_obs)
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(base_obs)
        
        # Calculate safety margin ratio
        min_distance = min(self.intruder_distance) if self.intruder_distance else float('inf')
        safety_margin_ratio = min_distance / self.intrusion_distance
        
        return {
            "observation_confidence": np.array([confidence], dtype=np.float64),
            "boundary_proximity": np.array([boundary_proximity]),
            "anomaly_score": np.array([anomaly_score]),
            "safety_margin_ratio": np.array([safety_margin_ratio])
        }

    def _calculate_observation_confidence(self):
        """Calculate confidence in observations based on historical consistency"""
        if len(self.observation_history) < 2:
            return 1.0
        
        # Simple consistency check - compare with previous observations
        current_distances = np.array(self.intruder_distance)
        prev_distances = self.observation_history[-1]["intruder_distance"] * WAYPOINT_DISTANCE_MAX
        
        # Calculate relative change
        relative_change = np.abs(current_distances - prev_distances) / (prev_distances + 1e-6)
        avg_change = np.mean(relative_change)
        
        # High changes indicate potential inconsistency
        confidence = max(0.0, 1.0 - avg_change)
        return confidence

    def _calculate_boundary_proximity(self, obs):
        """Calculate how close observations are to training data boundaries"""
        # Simple heuristic: check if values are in expected ranges
        distances = obs["intruder_distance"]
        
        # Check for extreme values that might indicate boundary conditions
        extreme_near = np.sum(distances < 0.1)  # Very close
        extreme_far = np.sum(distances > 0.9)   # Very far
        
        boundary_proximity = (extreme_near + extreme_far) / len(distances)
        return min(1.0, boundary_proximity)

    def _calculate_anomaly_score(self, obs):
        """Calculate anomaly score for potential hallucination detection"""
        # Simple anomaly detection based on statistical outliers
        if len(self.observation_history) < 3:
            return 0.0
        
        # Check for sudden jumps in intruder distances
        current_distances = obs["intruder_distance"]
        historical_distances = [h["intruder_distance"] for h in self.observation_history[-3:]]
        
        if len(historical_distances) > 0:
            historical_mean = np.mean(historical_distances, axis=0)
            historical_std = np.std(historical_distances, axis=0) + 1e-6
            
            z_scores = np.abs((current_distances - historical_mean) / historical_std)
            anomaly_score = np.mean(z_scores > 2.0)  # Values > 2 std deviations
            
            return min(1.0, anomaly_score)
        
        return 0.0

    def _detect_hallucinations(self, observation, action):
        """Detect potential ML hallucinations"""
        if not self.enable_hallucination_detection:
            return
        
        # Check for various hallucination indicators
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
        
        # 4. Safety margin violations
        safety_ratio = observation.get("safety_margin_ratio", [1.0])[0]
        if safety_ratio < 0.5:
            self.safety_margin_violations += 1
            hallucination_detected = True
        
        if hallucination_detected:
            self.hallucination_events.append({
                'step': len(self.observation_history),
                'confidence': observation.get("observation_confidence", [0.0])[0],
                'boundary_proximity': observation.get("boundary_proximity", [0.0])[0],
                'anomaly_score': observation.get("anomaly_score", [0.0])[0],
                'safety_ratio': safety_ratio,
                'action': action[0] if hasattr(action, '__len__') else action
            })

    def _get_info(self):
        """Get environment info including hallucination metrics"""
        base_info = {
            'total_reward': self.total_reward,
            'total_intrusions': self.total_intrusions,
            'average_drift': self.average_drift.mean() if len(self.average_drift) > 0 else 0.0,
            'safety_margin_level': self.safety_margin_level,
            'intrusion_distance': self.intrusion_distance
        }
        
        if self.enable_hallucination_detection:
            base_info.update({
                'hallucination_events': len(self.hallucination_events),
                'boundary_violations': self.boundary_violations,
                'safety_margin_violations': self.safety_margin_violations,
                'hallucination_rate': len(self.hallucination_events) / max(1, len(self.observation_history))
            })
        
        return base_info

    def _get_reward(self):
        """Calculate reward with hallucination penalties"""
        reach_reward = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_reward = self._check_intrusion()
        
        # Add hallucination penalty
        hallucination_reward = 0
        if self.enable_hallucination_detection and len(self.hallucination_events) > 0:
            recent_hallucinations = sum(1 for event in self.hallucination_events 
                                      if event['step'] >= len(self.observation_history) - 1)
            hallucination_reward = recent_hallucinations * HALLUCINATION_PENALTY

        total_reward = reach_reward + drift_reward + intrusion_reward + hallucination_reward
        self.total_reward += total_reward

        terminated = 0 not in self.wpt_reach if self.wpt_reach else False
        return total_reward, terminated
        
    def _check_waypoint(self):
        """Check waypoint reach status"""
        reward = 0
        for index, distance in enumerate(self.waypoint_distance):
            if distance < DISTANCE_MARGIN and self.wpt_reach[index] != 1:
                self.wpt_reach[index] = 1
                reward += REACH_REWARD
        return reward

    def _check_drift(self):
        """Check trajectory drift penalty"""
        if self.drift:
            drift = abs(np.deg2rad(self.drift[0]))
            self.average_drift = np.append(self.average_drift, drift)
            return drift * DRIFT_PENALTY
        return 0

    def _check_intrusion(self):
        """Check for intrusion violations using configured safety margin"""
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            _, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )
            if int_dis < self.intrusion_distance:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        
        return reward
    
    def _get_action(self, action):
        """Convert action to BlueSky heading command"""
        action_value = action[0] if hasattr(action, '__len__') else action
        new_heading = self.ac_hdg + action_value * D_HEADING
        bs.stack.stack(f"HDG KL001 {new_heading}")

    def _render_frame(self):
        """Render the environment visualization"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 200  # km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135, 206, 235))

        # Draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length) / max_distance) * self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length) / max_distance) * self.window_width

        pygame.draw.line(canvas, (0, 0, 0),
            (self.window_width/2 - heading_end_x/2, self.window_height/2 + heading_end_y/2),
            ((self.window_width/2) + heading_end_x/2, (self.window_height/2) - heading_end_y/2),
            width=4
        )

        # Draw intruders with safety margin indication
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            int_hdg = bs.traf.hdg[int_idx]
            
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )

            # Color based on safety margin
            if int_dis < self.intrusion_distance * 0.5:
                color = (255, 0, 0)      # Red - critical
            elif int_dis < self.intrusion_distance:
                color = (255, 165, 0)    # Orange - warning  
            else:
                color = (80, 80, 80)     # Gray - safe

            x_pos = (self.window_width/2) + (np.cos(np.deg2rad(int_qdr)) * (int_dis * NM2KM) / max_distance) * self.window_width
            y_pos = (self.window_height/2) - (np.sin(np.deg2rad(int_qdr)) * (int_dis * NM2KM) / max_distance) * self.window_height

            # Draw aircraft
            ac_length = 3
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * ac_length) / max_distance) * self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * ac_length) / max_distance) * self.window_width

            pygame.draw.line(canvas, color,
                (x_pos, y_pos),
                ((x_pos) + heading_end_x, (y_pos) - heading_end_y),
                width=4
            )

            # Draw safety margin circle
            pygame.draw.circle(canvas, color,
                (x_pos, y_pos),
                radius=(self.intrusion_distance * NM2KM / max_distance) * self.window_width,
                width=2
            )

        # Draw waypoints
        for qdr, dis, reach in zip(self.wpt_qdr, self.waypoint_distance, self.wpt_reach):
            circle_x = ((np.cos(np.deg2rad(qdr)) * dis) / max_distance) * self.window_width
            circle_y = ((np.sin(np.deg2rad(qdr)) * dis) / max_distance) * self.window_width

            color = (155, 155, 155) if reach else (255, 255, 255)

            pygame.draw.circle(canvas, color,
                ((self.window_width/2) + circle_x, (self.window_height/2) - circle_y),
                radius=4, width=0
            )
            
            pygame.draw.circle(canvas, color,
                ((self.window_width/2) + circle_x, (self.window_height/2) - circle_y),
                radius=(DISTANCE_MARGIN / max_distance) * self.window_width,
                width=2
            )

        # Draw hallucination indicator if enabled
        if self.enable_hallucination_detection and len(self.hallucination_events) > 0:
            recent_events = [e for e in self.hallucination_events if e['step'] >= len(self.observation_history) - 3]
            if recent_events:
                # Draw warning indicator
                pygame.draw.circle(canvas, (255, 255, 0),
                    (50, 50), radius=20, width=3
                )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        """Close the environment"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
