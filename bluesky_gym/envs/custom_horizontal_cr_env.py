# bluesky_gym/envs/custom_horizontal_cr_env.py
"""
Horizontal Conflict Resolution Environment for ML Hallucination Research

This environment is designed for researching ML-based hallucination effects on safety margins in air traffic control, specifically for horizontal conflict resolution scenarios.

Key features for thesis research:
- Parameterized training data envelope tracking
- Ground-truth conflict metrics (FP/FN calculation)
- Efficiency and intervention metrics
- Monte-Carlo stress scenario generator
- Severity-based hallucination penalties
- Comprehensive CSV logging schema
"""

import numpy as np
import pygame
from typing import Dict, Tuple, Optional, List, Any
import time

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
BASE_HALLUCINATION_PENALTY = -0.2

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

# Training data envelope bounds (derived from historical datasets)
DEFAULT_TRAINING_BOUNDS = {
    'intruder_distance': (0.0, 1.0),  # Normalized
    'cos_difference_pos': (-1.0, 1.0),
    'sin_difference_pos': (-1.0, 1.0),
    'x_difference_speed': (-1.0, 1.0),
    'y_difference_speed': (-1.0, 1.0),
    'waypoint_distance': (0.0, 1.0),
    'cos_drift': (-1.0, 1.0),
    'sin_drift': (-1.0, 1.0),
}

# Monte-Carlo complexity parameters
COMPLEXITY_SCENARIOS = {
    'nominal': {'dpsi_range': (45, 315), 'cpa_range': (3, 8), 'tlosh_range': (300, 1000)},
    'moderate': {'dpsi_range': (30, 330), 'cpa_range': (1, 6), 'tlosh_range': (150, 800)},
    'challenging': {'dpsi_range': (15, 345), 'cpa_range': (0.5, 4), 'tlosh_range': (80, 400)},
    'extreme': {'dpsi_range': (0, 360), 'cpa_range': (0.1, 2), 'tlosh_range': (50, 200)}
}

class CustomHorizontalCREnv(gym.Env):
    """
    Enhanced Horizontal Conflict Resolution Environment for ML Hallucination Research
    
    This environment builds upon the original HorizontalCREnv with additional features
    for studying ML hallucination effects and safety margin analysis.
    """
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 120}

    def __init__(self, render_mode=None, safety_margin='standard', 
                 enable_hallucination_detection=True, 
                 boundary_test_mode=False,
                 training_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 complexity_level: float = 0.0,
                 episode_id: Optional[str] = None):
        """
        Initialize the Enhanced Custom Horizontal CR Environment
        
        Args:
            render_mode: Rendering mode ('rgb_array', 'human', or None)
            safety_margin: Safety margin level ('conservative', 'standard', 'aggressive', 'critical')
            enable_hallucination_detection: Enable hallucination detection features
            boundary_test_mode: Enable boundary condition testing
            training_bounds: Dictionary of training data boundaries for each observation feature
            complexity_level: Complexity level for Monte-Carlo scenario generation (0.0-1.0)
            episode_id: Unique identifier for this episode
        """
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)
        
        # Configuration parameters
        self.safety_margin_level = safety_margin
        self.intrusion_distance = SAFETY_MARGIN_LEVELS[safety_margin]
        self.enable_hallucination_detection = enable_hallucination_detection
        self.boundary_test_mode = boundary_test_mode
        self.training_bounds = training_bounds or DEFAULT_TRAINING_BOUNDS
        self.complexity_level = max(0.0, min(1.0, complexity_level))  # Clamp to [0,1]
        self.episode_id = episode_id or f"ep_{int(time.time() * 1000)}"
        
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

        # Enhanced logging and analysis variables for thesis
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])
        self.hallucination_events = []
        self.boundary_violations = 0
        self.safety_margin_violations = 0
        
        # Ground-truth conflict tracking for FP/FN calculation
        self.ground_truth_conflicts = []
        self.agent_alerts = []
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        
        # Efficiency and intervention metrics
        self.intervention_count = 0
        self.cumulative_heading_deviation = 0.0
        self.last_heading = 0.0
        self.action_deadband = 0.1  # Minimum action magnitude to count as intervention
        
        # Envelope violation tracking
        self.envelope_violation_score = 0.0
        self.max_envelope_violation = 0.0
        
        # Observation history for hallucination detection
        self.observation_history = []
        self.max_history_length = 10
        
        # Episode performance metrics
        self.episode_start_time = time.time()
        self.timestep_count = 0
        
        # Rendering
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        
        bs.traf.reset()

        # Reset all logging variables
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])
        self.hallucination_events = []
        self.boundary_violations = 0
        self.safety_margin_violations = 0
        self.observation_history = []
        
        # Reset ground-truth conflict tracking
        self.ground_truth_conflicts = []
        self.agent_alerts = []
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        
        # Reset efficiency metrics
        self.intervention_count = 0
        self.cumulative_heading_deviation = 0.0
        self.last_heading = 0.0
        
        # Reset envelope tracking
        self.envelope_violation_score = 0.0
        self.max_envelope_violation = 0.0
        
        # Reset episode tracking
        self.episode_start_time = time.time()
        self.timestep_count = 0

        # Create ownship
        bs.traf.cre('KL001', actype="A320", acspd=AC_SPD)
        self.last_heading = bs.traf.hdg[0]

        # Generate conflicts using complexity-based parameters
        self._generate_conflicts()
        self._generate_waypoint()
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.timestep_count += 1
        
        # Track intervention and heading deviation
        self._track_intervention_metrics(action)
        
        self._get_action(action)

        # Execute multiple simulation steps per action
        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "human":
                observation = self._get_obs()
                self._render_frame()

        observation = self._get_obs()
        
        # Calculate ground truth conflicts and agent alerts
        self._calculate_ground_truth_conflicts()
        agent_alerted = self._check_agent_alert(observation, action)
        self._update_confusion_matrix(agent_alerted)
        
        reward, terminated = self._get_reward()
        
        # Detect potential hallucinations with severity-based penalties
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
        """Generate conflicting aircraft using Monte-Carlo complexity parameters"""
        target_idx = bs.traf.id2idx(acid)
        
        # Select scenario parameters based on complexity level
        scenario_keys = list(COMPLEXITY_SCENARIOS.keys())
        scenario_index = min(int(self.complexity_level * len(scenario_keys)), len(scenario_keys) - 1)
        scenario_name = scenario_keys[scenario_index]
        scenario_params = COMPLEXITY_SCENARIOS[scenario_name]
        
        for i in range(NUM_INTRUDERS):
            # Use complexity-based parameter ranges
            dpsi_min, dpsi_max = scenario_params['dpsi_range']
            cpa_min, cpa_max = scenario_params['cpa_range']
            tlosh_min, tlosh_max = scenario_params['tlosh_range']
            
            dpsi = np.random.uniform(dpsi_min, dpsi_max)
            cpa = np.random.uniform(cpa_min, cpa_max)
            tlosh = np.random.uniform(tlosh_min, tlosh_max)
            
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
        
        # Calculate boundary proximity (envelope violation)
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
        """Calculate envelope violation score based on training data boundaries"""
        violation_scores = []
        
        for feature_name, bounds in self.training_bounds.items():
            if feature_name in obs:
                feature_values = obs[feature_name]
                if isinstance(feature_values, np.ndarray):
                    feature_values = feature_values.flatten()
                else:
                    feature_values = [feature_values]
                
                min_bound, max_bound = bounds
                range_size = max_bound - min_bound
                
                for value in feature_values:
                    if value < min_bound:
                        violation = (min_bound - value) / range_size
                        violation_scores.append(violation)
                    elif value > max_bound:
                        violation = (value - max_bound) / range_size
                        violation_scores.append(violation)
                    else:
                        violation_scores.append(0.0)
        
        if violation_scores:
            self.envelope_violation_score = np.mean(violation_scores)
            self.max_envelope_violation = max(self.max_envelope_violation, self.envelope_violation_score)
            return min(1.0, self.envelope_violation_score)
        
        return 0.0

    def _calculate_anomaly_score(self, obs):
        """Calculate anomaly score for potential hallucination detection"""
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

    def _calculate_ground_truth_conflicts(self):
        """Calculate ground truth conflicts for FP/FN analysis"""
        ac_idx = bs.traf.id2idx('KL001')
        conflict_present = False
        
        # Check if any intruder violates separation within lookahead window
        lookahead_time = 60  # seconds
        
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            _, current_distance = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )
            
            # Simple conflict prediction (could be enhanced with trajectory prediction)
            if current_distance < self.intrusion_distance * 1.5:  # Expanded margin for prediction
                conflict_present = True
                break
        
        self.ground_truth_conflicts.append(conflict_present)

    def _check_agent_alert(self, observation, action):
        """Check if agent is alerting/intervening based on action magnitude"""
        action_magnitude = abs(action[0]) if hasattr(action, '__len__') else abs(action)
        agent_alerted = action_magnitude > self.action_deadband
        self.agent_alerts.append(agent_alerted)
        return agent_alerted

    def _update_confusion_matrix(self, agent_alerted):
        """Update confusion matrix for FP/FN calculation"""
        if len(self.ground_truth_conflicts) > 0:
            conflict_present = self.ground_truth_conflicts[-1]
            
            if conflict_present and agent_alerted:
                self.true_positives += 1
            elif conflict_present and not agent_alerted:
                self.false_negatives += 1
            elif not conflict_present and agent_alerted:
                self.false_positives += 1
            else:
                self.true_negatives += 1

    def _track_intervention_metrics(self, action):
        """Track efficiency and intervention metrics"""
        action_value = action[0] if hasattr(action, '__len__') else action
        
        # Count interventions
        if abs(action_value) > self.action_deadband:
            self.intervention_count += 1
        
        # Track heading deviation
        current_heading = bs.traf.hdg[0] if len(bs.traf.hdg) > 0 else self.last_heading
        heading_change = abs(current_heading - self.last_heading)
        if heading_change > 180:  # Handle angle wraparound
            heading_change = 360 - heading_change
        self.cumulative_heading_deviation += heading_change
        self.last_heading = current_heading

    def _detect_hallucinations(self, observation, action):
        """Detect potential ML hallucinations with severity-based penalties"""
        if not self.enable_hallucination_detection:
            return
        
        # Check for various hallucination indicators
        hallucination_detected = False
        severity_score = 0.0
        
        # 1. Inconsistent observations
        confidence = observation.get("observation_confidence", [1.0])[0]
        if confidence < 0.3:
            hallucination_detected = True
            severity_score += 1.0 - confidence
            
        # 2. Boundary violations (envelope violations)
        boundary_proximity = observation.get("boundary_proximity", [0.0])[0]
        if boundary_proximity > 0.7:
            self.boundary_violations += 1
            hallucination_detected = True
            severity_score += boundary_proximity
            
        # 3. High anomaly scores
        anomaly_score = observation.get("anomaly_score", [0.0])[0]
        if anomaly_score > 0.8:
            hallucination_detected = True
            severity_score += anomaly_score
        
        # 4. Safety margin violations
        safety_ratio = observation.get("safety_margin_ratio", [1.0])[0]
        if safety_ratio < 0.5:
            self.safety_margin_violations += 1
            hallucination_detected = True
            severity_score += 1.0 - safety_ratio
        
        if hallucination_detected:
            self.hallucination_events.append({
                'step': self.timestep_count,
                'confidence': confidence,
                'boundary_proximity': boundary_proximity,
                'anomaly_score': anomaly_score,
                'safety_ratio': safety_ratio,
                'severity_score': severity_score,
                'action': action[0] if hasattr(action, '__len__') else action,
                'episode_id': self.episode_id
            })

    def _get_info(self):
        """Get comprehensive environment info including all thesis metrics"""
        base_info = {
            # Basic environment info
            'episode_id': self.episode_id,
            'timestep': self.timestep_count,
            'episode_duration': time.time() - self.episode_start_time,
            'total_reward': self.total_reward,
            'total_intrusions': self.total_intrusions,
            'average_drift': self.average_drift.mean() if len(self.average_drift) > 0 else 0.0,
            
            # Safety margin configuration
            'safety_margin_level': self.safety_margin_level,
            'intrusion_distance': self.intrusion_distance,
            'complexity_level': self.complexity_level,
            
            # Ground-truth conflict metrics (FP/FN)
            'conflict_present': self.ground_truth_conflicts[-1] if self.ground_truth_conflicts else False,
            'alert_triggered': self.agent_alerts[-1] if self.agent_alerts else False,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            
            # Efficiency and intervention metrics
            'n_interventions': self.intervention_count,
            'cumulative_heading_deviation': self.cumulative_heading_deviation,
            'intervention_rate': self.intervention_count / max(1, self.timestep_count),
            
            # Training envelope and boundary metrics
            'envelope_violation_score': self.envelope_violation_score,
            'max_envelope_violation': self.max_envelope_violation,
        }
        
        if self.enable_hallucination_detection:
            base_info.update({
                'hallucination_events': len(self.hallucination_events),
                'boundary_violations': self.boundary_violations,
                'safety_margin_violations': self.safety_margin_violations,
                'hallucination_rate': len(self.hallucination_events) / max(1, self.timestep_count),
                
                # Calculate total severity-weighted hallucination penalty
                'total_hallucination_severity': sum(event['severity_score'] for event in self.hallucination_events),
                'avg_hallucination_severity': np.mean([event['severity_score'] for event in self.hallucination_events]) if self.hallucination_events else 0.0,
            })
        
        return base_info

    def _get_reward(self):
        """Calculate reward with severity-based hallucination penalties"""
        reach_reward = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_reward = self._check_intrusion()
        
        # Add severity-based hallucination penalty
        hallucination_reward = 0
        if self.enable_hallucination_detection and len(self.hallucination_events) > 0:
            # Get recent hallucination events from this timestep
            recent_events = [event for event in self.hallucination_events 
                           if event['step'] >= self.timestep_count]
            
            if recent_events:
                # Apply penalty proportional to severity
                total_severity = sum(event['severity_score'] for event in recent_events)
                hallucination_reward = BASE_HALLUCINATION_PENALTY * total_severity

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
        """Render the environment visualization with enhanced indicators"""
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

        # Color ownship based on hallucination/intervention status
        ownship_color = (0, 0, 0)  # Default black
        if self.enable_hallucination_detection and len(self.hallucination_events) > 0:
            recent_events = [e for e in self.hallucination_events if e['step'] >= self.timestep_count - 3]
            if recent_events:
                ownship_color = (255, 165, 0)  # Orange for hallucination warning

        pygame.draw.line(canvas, ownship_color,
            (self.window_width/2 - heading_end_x/2, self.window_height/2 + heading_end_y/2),
            ((self.window_width/2) + heading_end_x/2, (self.window_height/2) - heading_end_y/2),
            width=4
        )

        # Draw intruders with enhanced safety margin indication
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            int_hdg = bs.traf.hdg[int_idx]
            
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )

            # Enhanced color coding based on safety margins and complexity
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

        # Draw comprehensive status indicators
        if self.enable_hallucination_detection:
            # Hallucination indicator
            if len(self.hallucination_events) > 0:
                recent_events = [e for e in self.hallucination_events if e['step'] >= self.timestep_count - 5]
                if recent_events:
                    pygame.draw.circle(canvas, (255, 255, 0),
                        (50, 50), radius=20, width=3
                    )
                    font = pygame.font.Font(None, 24)
                    text = font.render("HALL", True, (255, 255, 0))
                    canvas.blit(text, (35, 42))
            
            # Boundary violation indicator
            if self.envelope_violation_score > 0.5:
                pygame.draw.circle(canvas, (255, 165, 0),
                    (100, 50), radius=20, width=3
                )
                font = pygame.font.Font(None, 24)
                text = font.render("BOUN", True, (255, 165, 0))
                canvas.blit(text, (85, 42))
            
            # Conflict detection indicator
            if self.ground_truth_conflicts and self.ground_truth_conflicts[-1]:
                pygame.draw.circle(canvas, (255, 0, 0),
                    (150, 50), radius=20, width=3
                )
                font = pygame.font.Font(None, 24)
                text = font.render("CONF", True, (255, 0, 0))
                canvas.blit(text, (135, 42))

        # Display comprehensive statistics
        if self.enable_hallucination_detection:
            font = pygame.font.Font(None, 20)
            
            # Row 1: Basic stats
            stats_text = [
                f"Episode: {self.episode_id}",
                f"Step: {self.timestep_count}",
                f"Complexity: {self.complexity_level:.2f}"
            ]
            
            # Row 2: Hallucination metrics
            stats_text.extend([
                f"Hallucinations: {len(self.hallucination_events)}",
                f"Boundary Violations: {self.boundary_violations}",
                f"Envelope Score: {self.envelope_violation_score:.3f}"
            ])
            
            # Row 3: Conflict detection metrics
            stats_text.extend([
                f"FP: {self.false_positives} FN: {self.false_negatives}",
                f"TP: {self.true_positives} TN: {self.true_negatives}",
                f"Interventions: {self.intervention_count}"
            ])
            
            # Display stats
            for i, text in enumerate(stats_text):
                rendered_text = font.render(text, True, (255, 255, 255))
                y_pos = self.window_height - 80 + (i % 3) * 20
                x_pos = 10 + (i // 3) * 200
                canvas.blit(rendered_text, (x_pos, y_pos))

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        """Close the environment"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
