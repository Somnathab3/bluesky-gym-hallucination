from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
from typing import Dict, List, Tuple, Optional, Any
import time
import pygame

# Import constants from your existing environments
DISTANCE_MARGIN = 5  # km
REACH_REWARD = 1
DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1
BASE_HALLUCINATION_PENALTY = -0.2

NUM_WAYPOINTS = 1
INTRUSION_DISTANCE = 5  # NM

WAYPOINT_DISTANCE_MIN = 100
WAYPOINT_DISTANCE_MAX = 150

D_HEADING = 45
AC_SPD = 150
NM2KM = 1.852
ACTION_FREQUENCY = 10

# Safety margin configurations (same as your custom envs)
SAFETY_MARGIN_LEVELS = {
    'conservative': 8.0,  # NM
    'standard': 5.0,      # NM  
    'aggressive': 3.0,    # NM
    'critical': 1.5       # NM for stress testing
}

# Same training bounds as your custom envs
DEFAULT_TRAINING_BOUNDS = {
    'intruder_distance': (0.0, 1.0),
    'cos_difference_pos': (-1.0, 1.0),
    'sin_difference_pos': (-1.0, 1.0),
    'x_difference_speed': (-1.0, 1.0),
    'y_difference_speed': (-1.0, 1.0),
    'waypoint_distance': (0.0, 1.0),
    'cos_drift': (-1.0, 1.0),
    'sin_drift': (-1.0, 1.0),
}

class PZMultiAgentCRConsistent(ParallelEnv):
    """
    PettingZoo Multi-Agent Environment Consistent with CustomHorizontalCREnv and CustomVerticalCREnv
    
    This environment maintains the same structure, observation spaces, action spaces, 
    and feature set as your existing custom environments while enabling multi-agent scenarios.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, 
                 num_agents=3,
                 render_mode=None, 
                 safety_margin='standard',
                 enable_hallucination_detection=True,
                 boundary_test_mode=False,
                 training_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 complexity_level: float = 0.0,
                 episode_id: Optional[str] = None,
                 curriculum_learning=False,
                 domain_randomization=True):
        """
        Initialize Multi-Agent Environment with same interface as custom envs
        """
        # Basic configuration (same as your custom envs)
        self.num_agents = max(2, min(6, num_agents))
        self.possible_agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agents = []
        self.render_mode = render_mode
        
        # Same configuration as your custom envs
        self.safety_margin_level = safety_margin
        self.intrusion_distance = SAFETY_MARGIN_LEVELS[safety_margin]
        self.enable_hallucination_detection = enable_hallucination_detection
        self.boundary_test_mode = boundary_test_mode
        self.training_bounds = training_bounds or DEFAULT_TRAINING_BOUNDS
        self.complexity_level = max(0.0, min(1.0, complexity_level))
        self.episode_id = episode_id or f"ma_ep_{int(time.time() * 1000)}"
        self.curriculum_learning = curriculum_learning
        self.domain_randomization = domain_randomization

        # Window configuration (same as horizontal env)
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)

        # Initialize BlueSky (same as your custom envs)
        bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 5;FF')

        # Define spaces (consistent with your custom envs)
        self._setup_spaces()
        
        # Internal state
        self._agent_callsigns = []
        self.timestep = 0
        self.episode_count = 0
        
        # Same tracking variables as your custom envs
        self._reset_tracking_variables()
        
        # Rendering
        self.window = None
        self.clock = None

    def _setup_spaces(self):
        """Setup spaces consistent with your custom environments"""
        # Same action space as your custom envs
        action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
        
        # Same observation structure as your custom envs
        obs_dict = {
            "intruder_distance": spaces.Box(-np.inf, np.inf, shape=(self.num_agents-1,), dtype=np.float64),
            "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape=(self.num_agents-1,), dtype=np.float64),
            "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape=(self.num_agents-1,), dtype=np.float64),
            "x_difference_speed": spaces.Box(-np.inf, np.inf, shape=(self.num_agents-1,), dtype=np.float64),
            "y_difference_speed": spaces.Box(-np.inf, np.inf, shape=(self.num_agents-1,), dtype=np.float64),
            "waypoint_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "cos_drift": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "sin_drift": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
        }
        
        # Same hallucination detection features as your custom envs
        if self.enable_hallucination_detection:
            obs_dict.update({
                "observation_confidence": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "boundary_proximity": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "anomaly_score": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
                "safety_margin_ratio": spaces.Box(0.0, 5.0, shape=(1,), dtype=np.float64),
            })
        
        observation_space = spaces.Dict(obs_dict)
        
        self.observation_spaces = {agent: observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: action_space for agent in self.possible_agents}

    def _reset_tracking_variables(self):
        """Reset tracking variables (same structure as your custom envs)"""
        # Same variables as your custom envs
        self.total_reward = {agent: 0 for agent in self.possible_agents}
        self.total_intrusions = {agent: 0 for agent in self.possible_agents}
        self.average_drift = {agent: np.array([]) for agent in self.possible_agents}
        self.hallucination_events = {agent: [] for agent in self.possible_agents}
        self.boundary_violations = {agent: 0 for agent in self.possible_agents}
        self.safety_margin_violations = {agent: 0 for agent in self.possible_agents}
        
        # Same ground-truth conflict tracking as your custom envs
        self.ground_truth_conflicts = {agent: [] for agent in self.possible_agents}
        self.agent_alerts = {agent: [] for agent in self.possible_agents}
        self.false_positives = {agent: 0 for agent in self.possible_agents}
        self.false_negatives = {agent: 0 for agent in self.possible_agents}
        self.true_positives = {agent: 0 for agent in self.possible_agents}
        self.true_negatives = {agent: 0 for agent in self.possible_agents}
        
        # Same temporal filtering as your enhanced envs
        self.conflict_history = {agent: [] for agent in self.possible_agents}
        self.alert_history = {agent: [] for agent in self.possible_agents}
        self.temporal_window = 3
        
        # Same efficiency metrics as your custom envs
        self.intervention_count = {agent: 0 for agent in self.possible_agents}
        self.cumulative_heading_deviation = {agent: 0.0 for agent in self.possible_agents}
        self.last_heading = {agent: 0.0 for agent in self.possible_agents}
        self.action_deadband = 0.1
        
        # Same envelope tracking as your custom envs
        self.envelope_violation_score = {agent: 0.0 for agent in self.possible_agents}
        self.max_envelope_violation = {agent: 0.0 for agent in self.possible_agents}
        
        # Same observation history as your custom envs
        self.observation_history = {agent: [] for agent in self.possible_agents}
        self.max_history_length = 10
        
        # Episode tracking
        self.episode_start_time = time.time()
        self.timestep_count = 0
        
        # Agent waypoints (same as your custom envs)
        self.wpt_lat = {agent: [] for agent in self.possible_agents}
        self.wpt_lon = {agent: [] for agent in self.possible_agents}
        self.wpt_reach = {agent: [] for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        """Reset environment (same structure as your custom envs)"""
        # Same reset logic as your custom envs
        bs.traf.reset()
        self.agents = self.possible_agents[:]
        self.timestep = 0
        self.episode_count += 1
        self._agent_callsigns = []
        
        self._reset_tracking_variables()
        
        # Apply curriculum learning (same as your enhanced envs)
        if self.curriculum_learning:
            self._update_curriculum_margin()
        
        # Create aircraft (similar to your custom envs)
        for i, agent in enumerate(self.agents):
            callsign = f"AC{i:02d}"
            self._agent_callsigns.append(callsign)
            bs.traf.cre(callsign, actype="A320", acspd=AC_SPD)
            self.last_heading[agent] = bs.traf.hdg[i]
        
        # Generate conflicts and waypoints (same method as your custom envs)
        self._generate_conflicts_and_waypoints()
        
        # Get initial observations
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._get_obs(i)
            
        return observations, {}

    def _update_curriculum_margin(self):
        """Same curriculum learning as your enhanced custom envs"""
        progress = min(1.0, self.episode_count / 1000)
        
        if self.safety_margin_level == 'standard':
            self.intrusion_distance = 3.0 + progress * (5.0 - 3.0)
        elif self.safety_margin_level == 'conservative':
            self.intrusion_distance = 5.0 + progress * (8.0 - 5.0)

    def _generate_conflicts_and_waypoints(self):
        """Generate conflicts and waypoints (adapted from your custom envs)"""
        # Generate conflicts for each agent with other agents as intruders
        for i, agent in enumerate(self.agents):
            callsign = self._agent_callsigns[i]
            target_idx = bs.traf.id2idx(callsign)
            
            # Generate waypoint (same as your custom envs)
            wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
            wpt_hdg_init = 0
            
            wpt_lat, wpt_lon = fn.get_point_at_distance(
                bs.traf.lat[target_idx], bs.traf.lon[target_idx], wpt_dis_init, wpt_hdg_init
            )
            
            self.wpt_lat[agent] = [wpt_lat]
            self.wpt_lon[agent] = [wpt_lon] 
            self.wpt_reach[agent] = [0]

    def step(self, actions):
        """Step function (same structure as your custom envs)"""
        self.timestep_count += 1
        
        # Track intervention metrics (same as your custom envs)
        for i, agent in enumerate(self.agents):
            self._track_intervention_metrics(agent, actions[agent])
        
        # Apply actions (same as your custom envs)
        for i, agent in enumerate(self.agents):
            self._get_action(i, actions[agent])

        # Execute simulation steps (same as your custom envs)
        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame()

        # Get observations
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._get_obs(i)
        
        # Calculate ground truth conflicts and update confusion matrix (same as your enhanced envs)
        for i, agent in enumerate(self.agents):
            self._calculate_ground_truth_conflicts_temporal(agent, i)
            agent_alerted = self._check_agent_alert(agent, actions[agent])
            self._update_confusion_matrix_temporal(agent, agent_alerted)
        
        # Calculate rewards (same structure as your custom envs)
        rewards = {}
        terminated = {}
        for i, agent in enumerate(self.agents):
            rewards[agent], terminated[agent] = self._get_reward(agent, i, actions[agent])
        
        # Detect hallucinations (same as your enhanced envs)
        if self.enable_hallucination_detection:
            for i, agent in enumerate(self.agents):
                self._detect_hallucinations(agent, observations[agent], actions[agent])

        # Get info (same structure as your custom envs)
        infos = {}
        for agent in self.agents:
            infos[agent] = self._get_info(agent)

        # Global termination condition
        global_terminated = any(terminated.values()) or self.timestep_count >= 500
        dones = {agent: global_terminated for agent in self.agents}
        truncateds = {agent: False for agent in self.agents}

        return observations, rewards, dones, truncateds, infos

    def _get_obs(self, agent_index):
        """Get observations (same structure as your custom envs)"""
        agent_id = self.agents[agent_index]
        callsign = self._agent_callsigns[agent_index]
        ac_idx = bs.traf.id2idx(callsign)

        # Same observation structure as your custom envs
        intruder_distance = []
        cos_bearing = []
        sin_bearing = []
        x_difference_speed = []
        y_difference_speed = []

        ac_hdg = bs.traf.hdg[ac_idx]

        # Process other agents as intruders (same logic as your custom envs)
        for j, other_callsign in enumerate(self._agent_callsigns):
            if j == agent_index:
                continue
                
            int_idx = bs.traf.id2idx(other_callsign)
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )
        
            intruder_distance.append(int_dis * NM2KM)

            bearing = ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            cos_bearing.append(np.cos(np.deg2rad(bearing)))
            sin_bearing.append(np.sin(np.deg2rad(bearing)))

            heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
            x_dif = -np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

            x_difference_speed.append(x_dif)
            y_difference_speed.append(y_dif)

        # Process waypoint information (same as your custom envs)
        waypoint_distance = []
        cos_drift = []
        sin_drift = []
        
        for lat, lon in zip(self.wpt_lat[agent_id], self.wpt_lon[agent_id]):
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon
            )
        
            waypoint_distance.append(wpt_dis * NM2KM)

            drift = ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)

            cos_drift.append(np.cos(np.deg2rad(drift)))
            sin_drift.append(np.sin(np.deg2rad(drift)))

        # Base observation (same structure as your custom envs)
        observation = {
            "intruder_distance": np.array(intruder_distance) / WAYPOINT_DISTANCE_MAX,
            "cos_difference_pos": np.array(cos_bearing),
            "sin_difference_pos": np.array(sin_bearing),
            "x_difference_speed": np.array(x_difference_speed) / AC_SPD,
            "y_difference_speed": np.array(y_difference_speed) / AC_SPD,
            "waypoint_distance": np.array(waypoint_distance) / WAYPOINT_DISTANCE_MAX,
            "cos_drift": np.array(cos_drift),
            "sin_drift": np.array(sin_drift)
        }
        
        # Add hallucination detection features (same as your custom envs)
        if self.enable_hallucination_detection:
            observation.update(self._get_hallucination_features(agent_id, observation))
        
        # Store observation history (same as your custom envs)
        self.observation_history[agent_id].append(observation.copy())
        if len(self.observation_history[agent_id]) > self.max_history_length:
            self.observation_history[agent_id].pop(0)
        
        return observation

    def _get_hallucination_features(self, agent_id, base_obs):
        """Same hallucination features as your custom envs"""
        confidence = self._calculate_observation_confidence(agent_id)
        boundary_proximity = self._calculate_boundary_proximity(agent_id, base_obs)
        anomaly_score = self._calculate_anomaly_score(agent_id, base_obs)
        
        # Calculate safety margin ratio
        if base_obs["intruder_distance"].size > 0:
            min_distance = np.min(base_obs["intruder_distance"]) * WAYPOINT_DISTANCE_MAX
            safety_margin_ratio = min_distance / self.intrusion_distance
        else:
            safety_margin_ratio = 5.0
        
        return {
            "observation_confidence": np.array([confidence], dtype=np.float64),
            "boundary_proximity": np.array([boundary_proximity]),
            "anomaly_score": np.array([anomaly_score]),
            "safety_margin_ratio": np.array([safety_margin_ratio])
        }

    # Include all the same helper methods from your custom envs:
    # _calculate_observation_confidence, _calculate_boundary_proximity, 
    # _calculate_anomaly_score, _track_intervention_metrics, etc.
    # (Same implementations as in your enhanced custom envs)

    def _calculate_observation_confidence(self, agent_id):
        """Same implementation as your custom envs"""
        if len(self.observation_history[agent_id]) < 2:
            return 1.0
        
        current_distances = self.observation_history[agent_id][-1]["intruder_distance"]
        prev_distances = self.observation_history[agent_id][-2]["intruder_distance"]
        
        if len(current_distances) > 0 and len(prev_distances) > 0:
            relative_change = np.abs(current_distances - prev_distances) / (prev_distances + 1e-6)
            avg_change = np.mean(relative_change)
            confidence = max(0.0, 1.0 - avg_change)
            return confidence
        
        return 1.0

    def _calculate_boundary_proximity(self, agent_id, obs):
        """Same implementation as your custom envs"""
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
            self.envelope_violation_score[agent_id] = np.mean(violation_scores)
            self.max_envelope_violation[agent_id] = max(
                self.max_envelope_violation[agent_id], 
                self.envelope_violation_score[agent_id]
            )
            return min(1.0, self.envelope_violation_score[agent_id])
        
        return 0.0

    def _calculate_anomaly_score(self, agent_id, obs):
        """Same implementation as your custom envs"""
        if len(self.observation_history[agent_id]) < 3:
            return 0.0
        
        current_distances = obs["intruder_distance"]
        historical_distances = [h["intruder_distance"] for h in self.observation_history[agent_id][-3:]]
        
        if len(historical_distances) > 0 and len(current_distances) > 0:
            historical_mean = np.mean(historical_distances, axis=0)
            historical_std = np.std(historical_distances, axis=0) + 1e-6
            
            z_scores = np.abs((current_distances - historical_mean) / historical_std)
            anomaly_score = np.mean(z_scores > 2.0)
            
            return min(1.0, anomaly_score)
        
        return 0.0

    def _track_intervention_metrics(self, agent_id, action):
        """Same implementation as your custom envs"""
        action_value = action[0] if hasattr(action, '__len__') else action
        
        if abs(action_value) > self.action_deadband:
            self.intervention_count[agent_id] += 1

    def _calculate_ground_truth_conflicts_temporal(self, agent_id, agent_index):
        """Same temporal filtering as your enhanced custom envs"""
        callsign = self._agent_callsigns[agent_index]
        ac_idx = bs.traf.id2idx(callsign)
        conflict_present = False
        
        # Check conflicts with other agents
        for j, other_callsign in enumerate(self._agent_callsigns):
            if j == agent_index:
                continue
                
            other_idx = bs.traf.id2idx(other_callsign)
            _, current_distance = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
                bs.traf.lat[other_idx], bs.traf.lon[other_idx]
            )
            
            if current_distance < self.intrusion_distance * 1.5:
                conflict_present = True
                break
        
        # Apply temporal filtering (same as your enhanced envs)
        self.conflict_history[agent_id].append(conflict_present)
        if len(self.conflict_history[agent_id]) > self.temporal_window:
            self.conflict_history[agent_id].pop(0)
        
        if len(self.conflict_history[agent_id]) >= self.temporal_window:
            filtered_conflict = sum(self.conflict_history[agent_id]) >= (self.temporal_window // 2 + 1)
        else:
            filtered_conflict = conflict_present
        
        self.ground_truth_conflicts[agent_id].append(filtered_conflict)

    def _check_agent_alert(self, agent_id, action):
        """Same implementation as your custom envs"""
        action_magnitude = abs(action[0]) if hasattr(action, '__len__') else abs(action)
        return action_magnitude > self.action_deadband

    def _update_confusion_matrix_temporal(self, agent_id, agent_alerted):
        """Same temporal filtering as your enhanced custom envs"""
        self.alert_history[agent_id].append(agent_alerted)
        if len(self.alert_history[agent_id]) > self.temporal_window:
            self.alert_history[agent_id].pop(0)
        
        if len(self.alert_history[agent_id]) >= self.temporal_window:
            filtered_alert = sum(self.alert_history[agent_id]) >= (self.temporal_window // 2 + 1)
        else:
            filtered_alert = agent_alerted
        
        self.agent_alerts[agent_id].append(filtered_alert)
        
        if len(self.ground_truth_conflicts[agent_id]) > 0:
            conflict_present = self.ground_truth_conflicts[agent_id][-1]
            
            if conflict_present and filtered_alert:
                self.true_positives[agent_id] += 1
            elif conflict_present and not filtered_alert:
                self.false_negatives[agent_id] += 1
            elif not conflict_present and filtered_alert:
                self.false_positives[agent_id] += 1
            else:
                self.true_negatives[agent_id] += 1

    def _get_reward(self, agent_id, agent_index, action):
        """Same reward structure as your custom envs"""
        reach_reward = self._check_waypoint(agent_id, agent_index)
        drift_reward = self._check_drift(agent_id, agent_index)
        intrusion_reward = self._check_intrusion(agent_id, agent_index)
        
        # Same hallucination penalty as your enhanced custom envs
        hallucination_reward = 0
        if self.enable_hallucination_detection and len(self.hallucination_events[agent_id]) > 0:
            recent_events = [event for event in self.hallucination_events[agent_id] 
                           if event['step'] >= self.timestep_count]
            
            if recent_events:
                action_magnitude = abs(action[0]) if hasattr(action, '__len__') else abs(action)
                total_severity = sum(event['severity_score'] for event in recent_events)
                disruption_factor = 1 + 2 * action_magnitude
                hallucination_reward = BASE_HALLUCINATION_PENALTY * total_severity * disruption_factor

        total_reward = reach_reward + drift_reward + intrusion_reward + hallucination_reward
        self.total_reward[agent_id] += total_reward

        terminated = 0 not in self.wpt_reach[agent_id] if self.wpt_reach[agent_id] else False
        return total_reward, terminated

    def _check_waypoint(self, agent_id, agent_index):
        """Same implementation as your custom envs"""
        callsign = self._agent_callsigns[agent_index]
        ac_idx = bs.traf.id2idx(callsign)
        reward = 0
        
        for i, (lat, lon) in enumerate(zip(self.wpt_lat[agent_id], self.wpt_lon[agent_id])):
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon
            )
            distance = wpt_dis * NM2KM
            
            if distance < DISTANCE_MARGIN and self.wpt_reach[agent_id][i] != 1:
                self.wpt_reach[agent_id][i] = 1
                reward += REACH_REWARD
        
        return reward

    def _check_drift(self, agent_id, agent_index):
        """Same implementation as your custom envs"""
        callsign = self._agent_callsigns[agent_index]
        ac_idx = bs.traf.id2idx(callsign)
        
        for lat, lon in zip(self.wpt_lat[agent_id], self.wpt_lon[agent_id]):
            wpt_qdr, _ = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon
            )
            
            drift = bs.traf.hdg[ac_idx] - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)
            
            drift_rad = abs(np.deg2rad(drift))
            self.average_drift[agent_id] = np.append(self.average_drift[agent_id], drift_rad)
            return drift_rad * DRIFT_PENALTY
        
        return 0

    def _check_intrusion(self, agent_id, agent_index):
        """Same implementation as your custom envs"""
        callsign = self._agent_callsigns[agent_index]
        ac_idx = bs.traf.id2idx(callsign)
        reward = 0
        
        for j, other_callsign in enumerate(self._agent_callsigns):
            if j == agent_index:
                continue
                
            other_idx = bs.traf.id2idx(other_callsign)
            _, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 
                bs.traf.lat[other_idx], bs.traf.lon[other_idx]
            )
            
            if int_dis < self.intrusion_distance:
                self.total_intrusions[agent_id] += 1
                reward += INTRUSION_PENALTY
        
        return reward

    def _get_action(self, agent_index, action):
        """Same action implementation as your custom envs"""
        callsign = self._agent_callsigns[agent_index]
        ac_idx = bs.traf.id2idx(callsign)
        action_value = action[0] if hasattr(action, '__len__') else action
        new_heading = bs.traf.hdg[ac_idx] + action_value * D_HEADING
        bs.stack.stack(f"HDG {callsign} {new_heading}")

    def _detect_hallucinations(self, agent_id, observation, action):
        """Same implementation as your enhanced custom envs"""
        if not self.enable_hallucination_detection:
            return
        
        hallucination_detected = False
        severity_score = 0.0
        
        confidence = observation.get("observation_confidence", [1.0])[0]
        if confidence < 0.3:
            hallucination_detected = True
            severity_score += 1.0 - confidence
            
        boundary_proximity = observation.get("boundary_proximity", [0.0])[0]
        if boundary_proximity > 0.7:
            self.boundary_violations[agent_id] += 1
            hallucination_detected = True
            severity_score += boundary_proximity
            
        anomaly_score = observation.get("anomaly_score", [0.0])[0]
        if anomaly_score > 0.8:
            hallucination_detected = True
            severity_score += anomaly_score
        
        safety_ratio = observation.get("safety_margin_ratio", [1.0])[0]
        if safety_ratio < 0.5:
            self.safety_margin_violations[agent_id] += 1
            hallucination_detected = True
            severity_score += 1.0 - safety_ratio
        
        if hallucination_detected:
            self.hallucination_events[agent_id].append({
                'step': self.timestep_count,
                'confidence': confidence,
                'boundary_proximity': boundary_proximity,
                'anomaly_score': anomaly_score,
                'safety_ratio': safety_ratio,
                'severity_score': severity_score,
                'action': action[0] if hasattr(action, '__len__') else action,
                'episode_id': self.episode_id
            })

    def _get_info(self, agent_id):
        """Same info structure as your custom envs"""
        return {
            'episode_id': self.episode_id,
            'timestep': self.timestep_count,
            'total_reward': self.total_reward[agent_id],
            'total_intrusions': self.total_intrusions[agent_id],
            'average_drift': self.average_drift[agent_id].mean() if len(self.average_drift[agent_id]) > 0 else 0.0,
            'safety_margin_level': self.safety_margin_level,
            'intrusion_distance': self.intrusion_distance,
            'complexity_level': self.complexity_level,
            'conflict_present': self.ground_truth_conflicts[agent_id][-1] if self.ground_truth_conflicts[agent_id] else False,
            'alert_triggered': self.agent_alerts[agent_id][-1] if self.agent_alerts[agent_id] else False,
            'false_positives': self.false_positives[agent_id],
            'false_negatives': self.false_negatives[agent_id],
            'true_positives': self.true_positives[agent_id],
            'true_negatives': self.true_negatives[agent_id],
            'n_interventions': self.intervention_count[agent_id],
            'intervention_rate': self.intervention_count[agent_id] / max(1, self.timestep_count),
            'envelope_violation_score': self.envelope_violation_score[agent_id],
            'max_envelope_violation': self.max_envelope_violation[agent_id],
            'hallucination_events': len(self.hallucination_events[agent_id]),
            'boundary_violations': self.boundary_violations[agent_id],
            'safety_margin_violations': self.safety_margin_violations[agent_id],
            'hallucination_rate': len(self.hallucination_events[agent_id]) / max(1, self.timestep_count),
            'total_hallucination_severity': sum(event['severity_score'] for event in self.hallucination_events[agent_id]),
            'avg_hallucination_severity': np.mean([event['severity_score'] for event in self.hallucination_events[agent_id]]) if self.hallucination_events[agent_id] else 0.0,
        }

    def _render_frame(self):
        """Same rendering style as your custom envs"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 200  # km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135, 206, 235))

        # Draw agents (same style as your custom envs)
        for i, callsign in enumerate(self._agent_callsigns):
            agent_id = self.agents[i]
            ac_idx = bs.traf.id2idx(callsign)
            
            if ac_idx >= 0:
                # Same rendering logic as your custom envs
                ac_length = 8
                x_pos = self.window_width/2
                y_pos = self.window_height/2
                
                heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length) / max_distance) * self.window_width
                heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length) / max_distance) * self.window_width

                # Color based on hallucination status (same as your custom envs)
                ownship_color = (0, 0, 0)
                if self.enable_hallucination_detection and len(self.hallucination_events[agent_id]) > 0:
                    recent_events = [e for e in self.hallucination_events[agent_id] if e['step'] >= self.timestep_count - 3]
                    if recent_events:
                        ownship_color = (255, 165, 0)

                pygame.draw.line(canvas, ownship_color,
                    (x_pos - heading_end_x/2, y_pos + heading_end_y/2),
                    (x_pos + heading_end_x/2, y_pos - heading_end_y/2),
                    width=4
                )
                
                # Agent label
                font = pygame.font.Font(None, 20)
                text = font.render(f"A{i}", True, (255, 255, 255))
                canvas.blit(text, (x_pos + 15, y_pos - 10))

        # Same status indicators as your custom envs
        if self.enable_hallucination_detection:
            font = pygame.font.Font(None, 20)
            stats_text = [
                f"Episode: {self.episode_id}",
                f"Step: {self.timestep_count}",
                f"Agents: {self.num_agents}",
                f"Complexity: {self.complexity_level:.2f}"
            ]
            
            for i, text in enumerate(stats_text):
                rendered_text = font.render(text, True, (255, 255, 255))
                y_pos = self.window_height - 80 + (i % 3) * 20
                x_pos = 10 + (i // 3) * 200
                canvas.blit(rendered_text, (x_pos, y_pos))

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def render(self):
        """Same render interface as your custom envs"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def close(self):
        """Same close method as your custom envs"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# Helper to convert to AEC env (same as original)
PZMultiAgentCRConsistentAEC = parallel_to_aec(PZMultiAgentCRConsistent)
