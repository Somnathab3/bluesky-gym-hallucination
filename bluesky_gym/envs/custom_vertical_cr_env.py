# bluesky_gym/envs/custom_vertical_cr_env.py
"""
Enhanced Custom Vertical Conflict Resolution Environment for ML Hallucination Research

This environment is designed for researching ML-based hallucination effects 
on safety margins in air traffic control, specifically for vertical 
conflict resolution scenarios during descent operations.

Key features for thesis research:
- Parameterized training data envelope tracking
- Ground-truth conflict metrics (FP/FN calculation)
- Efficiency and intervention metrics
- Monte-Carlo stress scenario generator
- Severity-based hallucination penalties
- Comprehensive CSV logging schema
- Enhanced hallucination reduction features
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
NM2KM = 1.852

INTRUSION_PENALTY = -50
ALT_DIF_REWARD_SCALE = -5/3000
CRASH_PENALTY = -100
RWY_ALT_DIF_REWARD_SCALE = -50/3000
BASE_HALLUCINATION_PENALTY = -0.2

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

# Training data envelope bounds for vertical domain (derived from historical datasets)
DEFAULT_TRAINING_BOUNDS = {
    'altitude': (-2.0, 2.0),  # Normalized altitude (std deviations)
    'vz': (-3.0, 3.0),        # Normalized vertical speed
    'target_altitude': (-2.0, 2.0),  # Normalized target altitude
    'runway_distance': (-2.0, 2.0),  # Normalized runway distance
    'intruder_distance': (0.0, 1.0), # Normalized distance
    'altitude_difference': (-2.0, 2.0),  # Normalized altitude difference
    'z_difference_speed': (-10.0, 10.0),  # Vertical speed difference
}

# Monte-Carlo complexity parameters for vertical scenarios
VERTICAL_COMPLEXITY_SCENARIOS = {
    'nominal': {'dpsi_range': (45, 315), 'cpa_range': (3, 8), 'tlosh_range': (300, 1000), 'dH_factor': 1.0},
    'moderate': {'dpsi_range': (30, 330), 'cpa_range': (1, 6), 'tlosh_range': (150, 800), 'dH_factor': 1.5},
    'challenging': {'dpsi_range': (15, 345), 'cpa_range': (0.5, 4), 'tlosh_range': (80, 400), 'dH_factor': 2.0},
    'extreme': {'dpsi_range': (0, 360), 'cpa_range': (0.1, 2), 'tlosh_range': (50, 200), 'dH_factor': 3.0}
}

class CustomVerticalCREnv(gym.Env):
    """
    Enhanced Vertical Conflict Resolution Environment for ML Hallucination Research
    
    This environment builds upon the original VerticalCREnv with additional features
    for studying ML hallucination effects and safety margin analysis in the vertical domain.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 120}

    def __init__(self, render_mode=None, vertical_safety_margin='standard', 
                 enable_hallucination_detection=True, 
                 boundary_test_mode=False,
                 stress_test_mode=False,
                 training_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 complexity_level: float = 0.0,
                 episode_id: Optional[str] = None,
                 curriculum_learning=False,
                 domain_randomization=True):
        """
        Initialize the Enhanced Custom Vertical CR Environment
        
        Args:
            render_mode: Rendering mode ('rgb_array', 'human', or None)
            vertical_safety_margin: Vertical safety margin level
            enable_hallucination_detection: Enable hallucination detection features
            boundary_test_mode: Enable boundary condition testing
            stress_test_mode: Enable stress testing with extreme scenarios
            training_bounds: Dictionary of training data boundaries for each observation feature
            complexity_level: Complexity level for Monte-Carlo scenario generation (0.0-1.0)
            episode_id: Unique identifier for this episode
            curriculum_learning: Enable curriculum learning for safety margins
            domain_randomization: Enable domain randomization for robust training
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
        self.training_bounds = training_bounds or DEFAULT_TRAINING_BOUNDS
        self.complexity_level = max(0.0, min(1.0, complexity_level))  # Clamp to [0,1]
        self.episode_id = episode_id or f"ep_{int(time.time() * 1000)}"
        self.curriculum_learning = curriculum_learning
        self.domain_randomization = domain_randomization
        
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

        # Enhanced logging and analysis variables for thesis
        self.total_reward = 0
        self.total_intrusions = 0
        self.final_altitude = 0
        self.hallucination_events = []
        self.boundary_violations = 0
        self.vertical_safety_violations = 0
        self.extreme_descent_events = 0
        
        # Ground-truth conflict tracking for FP/FN calculation with temporal filtering
        self.ground_truth_conflicts = []
        self.agent_alerts = []
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        
        # Temporal filtering for conflict detection
        self.conflict_history = []
        self.alert_history = []
        self.temporal_window = 3  # Require conflicts to persist for 3 timesteps
        
        # Efficiency and intervention metrics
        self.intervention_count = 0
        self.cumulative_descent_deviation = 0.0
        self.last_vertical_speed = 0.0
        self.action_deadband = 0.2  # Minimum action magnitude to count as intervention
        
        # Envelope violation tracking
        self.envelope_violation_score = 0.0
        self.max_envelope_violation = 0.0
        
        # Observation history for hallucination detection
        self.observation_history = []
        self.max_history_length = 10
        
        # Trajectory prediction for improved hallucination detection
        self.predicted_trajectories = {}
        self.actual_trajectories = {}
        
        # Target altitude for current episode
        self.target_alt = None
        
        # Episode performance metrics
        self.episode_start_time = time.time()
        self.timestep_count = 0
        self.episode_count = 0
        
        # Curriculum learning progression
        if self.curriculum_learning:
            self.base_vertical_margin = self.vertical_margin
        
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
        self.final_altitude = 0
        self.hallucination_events = []
        self.boundary_violations = 0
        self.vertical_safety_violations = 0
        self.extreme_descent_events = 0
        self.observation_history = []
        
        # Reset ground-truth conflict tracking
        self.ground_truth_conflicts = []
        self.agent_alerts = []
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        
        # Reset temporal filtering
        self.conflict_history = []
        self.alert_history = []
        
        # Reset efficiency metrics
        self.intervention_count = 0
        self.cumulative_descent_deviation = 0.0
        self.last_vertical_speed = 0.0
        
        # Reset envelope tracking
        self.envelope_violation_score = 0.0
        self.max_envelope_violation = 0.0
        
        # Reset trajectory tracking
        self.predicted_trajectories = {}
        self.actual_trajectories = {}
        
        # Reset episode tracking
        self.episode_start_time = time.time()
        self.timestep_count = 0
        self.episode_count += 1
        
        # Apply curriculum learning for vertical safety margins
        if self.curriculum_learning:
            self._update_curriculum_vertical_margin()

        # Set initial and target altitudes based on complexity and stress testing
        if self.stress_test_mode:
            # More challenging altitude scenarios for stress testing
            alt_init = np.random.randint(ALT_MAX - 500, ALT_MAX)
            self.target_alt = np.random.randint(ALT_MIN, alt_init - 1000)
        else:
            alt_init = np.random.randint(ALT_MIN, ALT_MAX)
            self.target_alt = alt_init + np.random.randint(-TARGET_ALT_DIF, TARGET_ALT_DIF)

        # Add domain randomization to target altitude
        if self.domain_randomization:
            altitude_noise = np.random.normal(0, 200)  # ±200m noise
            alt_init += altitude_noise
            self.target_alt += altitude_noise * 0.5  # Smaller noise for target

        # Create ownship
        bs.traf.cre('KL001', actype="A320", acalt=alt_init, acspd=AC_SPD)
        bs.traf.swvnav[0] = False
        self.last_vertical_speed = bs.traf.vs[0]

        # Generate conflicts using complexity-based parameters
        self._generate_conflicts(acid='KL001')

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.timestep_count += 1
        
        # Track intervention and descent deviation
        self._track_intervention_metrics(action)
        
        # Store predicted trajectory before taking action
        if self.enable_hallucination_detection:
            self._predict_vertical_trajectories(action)
        
        self._get_action(action)

        # Execute multiple simulation steps per action
        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame()
                observation = self._get_obs()

        observation = self._get_obs()
        
        # Update actual trajectories for comparison
        if self.enable_hallucination_detection:
            self._update_actual_vertical_trajectories()
        
        # Calculate ground truth conflicts with temporal filtering
        self._calculate_ground_truth_conflicts_temporal()
        agent_alerted = self._check_agent_alert(observation, action)
        self._update_confusion_matrix_temporal(agent_alerted)
        
        reward, terminated = self._get_reward(action)
        
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
        """Generate conflicting aircraft with enhanced boundary testing and domain randomization for vertical scenarios"""
        target_idx = bs.traf.id2idx(acid)
        altitude = bs.traf.alt[target_idx]
        spd = bs.traf.gs[target_idx]
        
        # Select scenario parameters based on complexity level
        scenario_keys = list(VERTICAL_COMPLEXITY_SCENARIOS.keys())
        scenario_index = min(int(self.complexity_level * len(scenario_keys)), len(scenario_keys) - 1)
        scenario_name = scenario_keys[scenario_index]
        scenario_params = VERTICAL_COMPLEXITY_SCENARIOS[scenario_name]
        
        # Domain randomization: Add environmental noise
        altitude_turbulence = np.random.normal(0, 50) if self.domain_randomization else 0
        speed_noise_factor = np.random.uniform(0.95, 1.05) if self.domain_randomization else 1.0
        
        for i in range(NUM_INTRUDERS):
            # Use complexity-based parameter ranges
            dpsi_min, dpsi_max = scenario_params['dpsi_range']
            cpa_min, cpa_max = scenario_params['cpa_range']
            tlosh_min, tlosh_max = scenario_params['tlosh_range']
            dH_factor = scenario_params['dH_factor']
            
            dpsi = np.random.uniform(dpsi_min, dpsi_max)
            cpa = np.random.uniform(cpa_min, cpa_max)
            tlosh = np.random.uniform(tlosh_min, tlosh_max)
            
            # Add domain randomization noise
            if self.domain_randomization:
                dpsi += np.random.normal(0, 5)  # Angular noise
                cpa += np.random.normal(0, 0.5)  # Distance noise
                tlosh += np.random.normal(0, 50)  # Time noise
                
                # Ensure bounds
                dpsi = np.clip(dpsi, 0, 360)
                cpa = max(0.1, cpa)
                tlosh = max(50, tlosh)
            
            # Enhanced altitude conflict generation with complexity scaling
            average_tod = (DEFAULT_RWY_DIS * 1000 / spd) - 2 * self.target_alt / ACTION_2_MS
            
            if self.stress_test_mode or self.complexity_level > 0.7:
                # Create more challenging vertical conflicts
                if tlosh > average_tod:
                    dH_range = int((self.target_alt - altitude) * dH_factor + 200)
                    dH = np.random.randint(int(-altitude + 200), dH_range)
                else:
                    dH_range = int(200 * dH_factor)
                    dH = np.random.randint(int((self.target_alt - altitude) - dH_range), 
                                         int((self.target_alt - altitude) + dH_range))
            else:
                if tlosh > average_tod:
                    dH = np.random.randint(int(-altitude + 500), int((self.target_alt - altitude) + 100))
                else:
                    dH = np.random.randint(int((self.target_alt - altitude) - 500), 
                                         int((self.target_alt - altitude) + 500))
            
            # Add domain randomization to altitude differences
            if self.domain_randomization:
                dH += np.random.normal(0, 100)  # ±100m altitude noise
            
            tlosv = 100000000000.

            bs.traf.creconfs(acid=f'{i}', actype="A320", targetidx=target_idx,
                           dpsi=dpsi, dcpa=cpa, tlosh=tlosh, dH=dH, tlosv=tlosv)
            bs.traf.alt[i+1] = bs.traf.alt[target_idx] + dH + altitude_turbulence
            bs.traf.ap.selaltcmd(i+1, bs.traf.alt[target_idx] + dH + altitude_turbulence, 0)
            
            # Apply speed noise
            if self.domain_randomization:
                bs.traf.gs[i+1] *= speed_noise_factor

    def _update_curriculum_vertical_margin(self):
        """Update vertical safety margin based on curriculum learning progression"""
        # Start with aggressive margin, gradually increase to target
        progress = min(1.0, self.episode_count / 1000)  # Full curriculum over 1000 episodes
        
        if self.vertical_safety_margin_level == 'standard':
            # Start at aggressive (500 ft), progress to standard (1000 ft)
            aggressive_margin = 500 * 0.3048
            standard_margin = 1000 * 0.3048
            self.vertical_margin = aggressive_margin + progress * (standard_margin - aggressive_margin)
        elif self.vertical_safety_margin_level == 'conservative':
            # Start at standard (1000 ft), progress to conservative (1500 ft)
            standard_margin = 1000 * 0.3048
            conservative_margin = 1500 * 0.3048
            self.vertical_margin = standard_margin + progress * (conservative_margin - standard_margin)

    def _predict_vertical_trajectories(self, action):
        """Predict vertical trajectories for hallucination detection"""
        ac_idx = bs.traf.id2idx('KL001')
        
        # Simple vertical trajectory prediction for ownship
        action_value = action[0] if hasattr(action, '__len__') else action
        action_ms = action_value * ACTION_2_MS
        
        # Predict altitude change
        time_step = ACTION_FREQUENCY * 1  # 1 second per simulation step
        predicted_altitude = self.altitude + action_ms * time_step
        
        # Store prediction
        self.predicted_trajectories[self.timestep_count] = {
            'ownship_altitude': predicted_altitude,
            'ownship_vz': action_ms,
            'intruder_altitudes': []
        }
        
        # Predict intruder altitudes (simple linear extrapolation)
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            if int_idx < len(bs.traf.alt):
                current_alt = bs.traf.alt[int_idx]
                current_vz = bs.traf.vs[int_idx]
                
                # Predict altitude after ACTION_FREQUENCY simulation steps
                predicted_int_altitude = current_alt + current_vz * time_step * ACTION_FREQUENCY
                
                self.predicted_trajectories[self.timestep_count]['intruder_altitudes'].append(predicted_int_altitude)

    def _update_actual_vertical_trajectories(self):
        """Update actual vertical trajectories for comparison with predictions"""
        ac_idx = bs.traf.id2idx('KL001')
        
        self.actual_trajectories[self.timestep_count] = {
            'ownship_altitude': bs.traf.alt[ac_idx],
            'ownship_vz': bs.traf.vs[ac_idx],
            'intruder_altitudes': []
        }
        
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            if int_idx < len(bs.traf.alt):
                self.actual_trajectories[self.timestep_count]['intruder_altitudes'].append(bs.traf.alt[int_idx])

    def _calculate_ground_truth_conflicts_temporal(self):
        """Calculate ground truth conflicts with temporal filtering for vertical domain"""
        ac_idx = bs.traf.id2idx('KL001')
        conflict_present = False
        
        # Check if any intruder violates vertical separation within lookahead window
        lookahead_time = 60  # seconds
        
        for i in range(NUM_INTRUDERS):
            int_idx = i + 1
            _, horizontal_distance = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )
            vertical_distance = abs(bs.traf.alt[ac_idx] - bs.traf.alt[int_idx])
            
            # Check for potential vertical conflicts
            if (horizontal_distance < INTRUSION_DISTANCE and 
                vertical_distance < self.vertical_margin):  # Not Expanded margin for false prediction
                conflict_present = True
                break
        
        # Add to conflict history
        self.conflict_history.append(conflict_present)
        if len(self.conflict_history) > self.temporal_window:
            self.conflict_history.pop(0)
        
        # Determine filtered conflict (require persistence)
        if len(self.conflict_history) >= self.temporal_window:
            # Require majority of recent timesteps to indicate conflict
            filtered_conflict = sum(self.conflict_history) >= (self.temporal_window // 2 + 1)
        else:
            filtered_conflict = conflict_present
        
        self.ground_truth_conflicts.append(filtered_conflict)

    def _update_confusion_matrix_temporal(self, agent_alerted):
        """Update confusion matrix with temporal filtering for FP/FN calculation"""
        # Add to alert history
        self.alert_history.append(agent_alerted)
        if len(self.alert_history) > self.temporal_window:
            self.alert_history.pop(0)
        
        # Determine filtered alert (require persistence for positive classification)
        if len(self.alert_history) >= self.temporal_window:
            filtered_alert = sum(self.alert_history) >= (self.temporal_window // 2 + 1)
        else:
            filtered_alert = agent_alerted
        
        self.agent_alerts.append(filtered_alert)
        
        if len(self.ground_truth_conflicts) > 0:
            conflict_present = self.ground_truth_conflicts[-1]
            
            if conflict_present and filtered_alert:
                self.true_positives += 1
            elif conflict_present and not filtered_alert:
                self.false_negatives += 1
            elif not conflict_present and filtered_alert:
                self.false_positives += 1
            else:
                self.true_negatives += 1

    def _check_vertical_trajectory_prediction_error(self):
        """Check for vertical prediction errors that might indicate hallucinations"""
        if len(self.predicted_trajectories) < 2 or len(self.actual_trajectories) < 2:
            return 0.0
        
        prediction_error = 0.0
        comparison_steps = min(3, len(self.predicted_trajectories))
        
        for i in range(1, comparison_steps + 1):
            step = self.timestep_count - i
            if step in self.predicted_trajectories and step in self.actual_trajectories:
                pred = self.predicted_trajectories[step]
                actual = self.actual_trajectories[step]
                
                # Compare altitude prediction error
                altitude_error = abs(pred['ownship_altitude'] - actual['ownship_altitude'])
                normalized_alt_error = altitude_error / 500.0  # Normalize by 500m
                
                # Compare vertical speed prediction error
                vz_error = abs(pred['ownship_vz'] - actual['ownship_vz'])
                normalized_vz_error = vz_error / 15.0  # Normalize by 15 m/s
                
                prediction_error += (normalized_alt_error + normalized_vz_error) / 2.0
        
        return min(1.0, prediction_error / comparison_steps)

    def _get_reward(self, action):
        """Calculate reward with enhanced severity-based hallucination penalties"""
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
            
        # Enhanced severity-based hallucination penalty
        hallucination_reward = 0
        if self.enable_hallucination_detection and len(self.hallucination_events) > 0:
            # Get recent hallucination events from this timestep
            recent_events = [event for event in self.hallucination_events 
                           if event['step'] >= self.timestep_count]
            
            if recent_events:
                # Apply penalty proportional to severity AND action magnitude
                action_magnitude = abs(action[0]) if hasattr(action, '__len__') else abs(action)
                total_severity = sum(event['severity_score'] for event in recent_events)
                
                # Scale penalty by action magnitude for vertical maneuvers
                disruption_factor = 1 + 3 * action_magnitude  # Factor between 1-4 (higher for vertical)
                hallucination_reward = BASE_HALLUCINATION_PENALTY * total_severity * disruption_factor
            
        reward = alt_penalty + int_penalty + hallucination_reward
        self.total_reward += reward
        return reward, done

    # [Include all the standard methods from the original with these key enhancements:]
    
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
        """Calculate envelope violation score based on training data boundaries for vertical domain"""
        violation_scores = []
        
        for feature_name, bounds in self.training_bounds.items():
            if feature_name in obs:
                feature_values = obs[feature_name]
                if isinstance(feature_values, np.ndarray):
                    feature_values = feature_values.flatten()
                else:
                    feature_values = [feature_values]
                
                min_bound, max_bound = bounds
                range_size = max_bound - min_bound if max_bound != min_bound else 1.0
                
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

    def _check_agent_alert(self, observation, action):
        """Check if agent is alerting/intervening based on action magnitude"""
        action_magnitude = abs(action[0]) if hasattr(action, '__len__') else abs(action)
        agent_alerted = action_magnitude > self.action_deadband
        return agent_alerted

    def _track_intervention_metrics(self, action):
        """Track efficiency and intervention metrics for vertical domain"""
        action_value = action[0] if hasattr(action, '__len__') else action
        
        # Count interventions
        if abs(action_value) > self.action_deadband:
            self.intervention_count += 1
        
        # Track descent rate deviation
        current_vz = bs.traf.vs[0] if len(bs.traf.vs) > 0 else self.last_vertical_speed
        vz_change = abs(current_vz - self.last_vertical_speed)
        self.cumulative_descent_deviation += vz_change
        self.last_vertical_speed = current_vz

    def _detect_hallucinations(self, observation, action):
        """Detect potential ML hallucinations in vertical domain with enhanced severity-based penalties"""
        if not self.enable_hallucination_detection:
            return
        
        hallucination_detected = False
        severity_score = 0.0
        
        # 1. Inconsistent observations
        confidence = observation.get("observation_confidence", [1.0])[0]
        if confidence < 0.3:
            hallucination_detected = True
            severity_score += 1.0 - confidence
            
        # 2. Boundary violations
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
        
        # 4. Vertical safety margin violations
        vertical_safety_ratio = observation.get("vertical_safety_ratio", [1.0])[0]
        if vertical_safety_ratio < 0.3:
            self.vertical_safety_violations += 1
            hallucination_detected = True
            severity_score += 1.0 - vertical_safety_ratio
        
        # 5. Descent rate anomalies
        descent_rate_anomaly = observation.get("descent_rate_anomaly", [0.0])[0]
        if descent_rate_anomaly > 0.5:
            hallucination_detected = True
            severity_score += descent_rate_anomaly
            
        # 6. Altitude boundary risks
        altitude_boundary_risk = observation.get("altitude_boundary_risk", [0.0])[0]
        if altitude_boundary_risk > 0.8:
            hallucination_detected = True
            severity_score += altitude_boundary_risk
            
        # 7. Vertical trajectory prediction errors
        prediction_error = self._check_vertical_trajectory_prediction_error()
        if prediction_error > 0.6:
            hallucination_detected = True
            severity_score += prediction_error
        
        if hallucination_detected:
            self.hallucination_events.append({
                'step': self.timestep_count,
                'confidence': confidence,
                'boundary_proximity': boundary_proximity,
                'anomaly_score': anomaly_score,
                'vertical_safety_ratio': vertical_safety_ratio,
                'descent_rate_anomaly': descent_rate_anomaly,
                'altitude_boundary_risk': altitude_boundary_risk,
                'prediction_error': prediction_error,
                'severity_score': severity_score,
                'action': action[0] if hasattr(action, '__len__') else action,
                'altitude': self.altitude,
                'vertical_speed': self.vz,
                'episode_id': self.episode_id
            })

    def _get_info(self):
        """Get comprehensive environment info including all thesis metrics"""
        base_info = {
            # Basic environment info
            'episode_id': self.episode_id,
            'timestep': self.timestep_count,
            'episode_duration': time.time() - self.episode_start_time,
            "total_reward": self.total_reward,
            "total_intrusions": self.total_intrusions,
            "final_altitude": self.final_altitude,
            
            # Safety margin configuration
            "vertical_safety_margin_level": self.vertical_safety_margin_level,
            "vertical_margin": self.vertical_margin,
            "target_altitude": self.target_alt,
            "current_altitude": self.altitude,
            "vertical_speed": self.vz,
            "complexity_level": self.complexity_level,
            "curriculum_learning": self.curriculum_learning,
            "domain_randomization": self.domain_randomization,
            
            # Ground-truth conflict metrics (FP/FN) with temporal filtering
            'conflict_present': self.ground_truth_conflicts[-1] if self.ground_truth_conflicts else False,
            'alert_triggered': self.agent_alerts[-1] if self.agent_alerts else False,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            
            # Efficiency and intervention metrics
            'n_interventions': self.intervention_count,
            'cumulative_descent_deviation': self.cumulative_descent_deviation,
            'intervention_rate': self.intervention_count / max(1, self.timestep_count),
            
            # Training envelope and boundary metrics
            'envelope_violation_score': self.envelope_violation_score,
            'max_envelope_violation': self.max_envelope_violation,
        }
        
        if self.enable_hallucination_detection:
            base_info.update({
                'hallucination_events': len(self.hallucination_events),
                'boundary_violations': self.boundary_violations,
                'vertical_safety_violations': self.vertical_safety_violations,
                'extreme_descent_events': self.extreme_descent_events,
                'hallucination_rate': len(self.hallucination_events) / max(1, self.timestep_count),
                
                # Calculate total severity-weighted hallucination penalty
                'total_hallucination_severity': sum(event['severity_score'] for event in self.hallucination_events),
                'avg_hallucination_severity': np.mean([event['severity_score'] for event in self.hallucination_events]) if self.hallucination_events else 0.0,
            })
        
        return base_info
    
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
        """Render the environment visualization with comprehensive hallucination indicators"""
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
            recent_events = [e for e in self.hallucination_events if e['step'] >= self.timestep_count - 3]
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
                  vertical_separation < self.vertical_margin):
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

        # Draw comprehensive hallucination and safety indicators
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
                
            # Extreme descent indicator
            if self.extreme_descent_events > 0:
                pygame.draw.circle(canvas, (128, 0, 128),
                    (200, 50), radius=20, width=3
                )
                font = pygame.font.Font(None, 24)
                text = font.render("DESC", True, (128, 0, 128))
                canvas.blit(text, (185, 42))

        # Display comprehensive statistics
        if self.enable_hallucination_detection:
            font = pygame.font.Font(None, 18)
            
            # Row 1: Basic episode info
            stats_text = [
                f"Episode: {self.episode_id[-8:]}",  # Show last 8 chars
                f"Step: {self.timestep_count}",
                f"Complexity: {self.complexity_level:.2f}"
            ]
            
            # Row 2: Hallucination metrics
            stats_text.extend([
                f"Hallucinations: {len(self.hallucination_events)}",
                f"Boundary Viol: {self.boundary_violations}",
                f"Envelope: {self.envelope_violation_score:.3f}"
            ])
            
            # Row 3: Conflict detection metrics
            stats_text.extend([
                f"FP: {self.false_positives} FN: {self.false_negatives}",
                f"TP: {self.true_positives} TN: {self.true_negatives}",
                f"Interventions: {self.intervention_count}"
            ])
            
            # Row 4: Altitude and descent metrics
            stats_text.extend([
                f"Alt: {int(self.altitude)}m VS: {self.vz:.1f}m/s",
                f"Target: {int(self.target_alt)}m",
                f"Safety: {self.vertical_safety_margin_level}"
            ])
            
            # Display stats in a compact grid
            for i, text in enumerate(stats_text):
                rendered_text = font.render(text, True, (255, 255, 255))
                y_pos = self.window_height - 80 + (i % 4) * 18
                x_pos = 5 + (i // 4) * 170
                canvas.blit(rendered_text, (x_pos, y_pos))

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        """Close the environment"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
