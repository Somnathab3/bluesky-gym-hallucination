import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from typing import Dict, List, Any, Optional

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

# Constants from your existing environments
DISTANCE_MARGIN = 5  # km
NM2KM = 1.852
INTRUSION_PENALTY = -50
ALT_DIF_REWARD_SCALE = -5/3000
CRASH_PENALTY = -100
RWY_ALT_DIF_REWARD_SCALE = -50/3000
DRIFT_PENALTY = -0.1
REACH_REWARD = 1

# Environment configuration
NUM_AIRCRAFT = 6  # Number of aircraft agents
INTRUSION_DISTANCE = 5  # NM
VERTICAL_MARGIN = 1000 * 0.3048  # ft converted to meters
ACTION_2_MS = 12.5  # vertical speed conversion
D_HEADING = 45  # degrees
D_SPEED = 20  # knots
AC_SPD = 150  # knots
ACTION_FREQUENCY = 30

# Altitude and target parameters
ALT_MEAN = 1500
ALT_STD = 3000
VZ_MEAN = 0
VZ_STD = 5
RWY_DIS_MEAN = 100
RWY_DIS_STD = 200
DEFAULT_RWY_DIS = 200
RWY_LAT = 52
RWY_LON = 4

ALT_MIN = 2000
ALT_MAX = 4000
TARGET_ALT_DIF = 500

class MARLConflictResolutionEnv(ParallelEnv):
    """
    Multi-Agent Reinforcement Learning Environment for Air Traffic Conflict Resolution
    
    This environment combines vertical and horizontal conflict resolution in a multi-agent setting.
    Each aircraft is controlled by an independent agent that must avoid conflicts with other aircraft
    while reaching their destination efficiently.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, 
                 num_aircraft: int = NUM_AIRCRAFT,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 2000,
                 enable_communication: bool = False,
                 cooperative_reward: bool = True):
        """
        Initialize the MARL conflict resolution environment
        
        Args:
            num_aircraft: Number of aircraft agents
            render_mode: Rendering mode ("human", "rgb_array", or None)
            max_episode_steps: Maximum steps per episode
            enable_communication: Whether to enable inter-agent communication
            cooperative_reward: Whether to use cooperative reward structure
        """
        super().__init__()
        
        self.num_aircraft = num_aircraft
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.enable_communication = enable_communication
        self.cooperative_reward = cooperative_reward
        
        # Initialize BlueSky
        bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')
        
        # Agent management
        self.possible_agents = [f"aircraft_{i}" for i in range(num_aircraft)]
        self.agents = self.possible_agents.copy()
        
        # Environment state
        self.current_step = 0
        self.aircraft_data = {}
        self.conflicts = []
        self.total_intrusions = 0
        self.episode_rewards = {agent: 0.0 for agent in self.agents}
        
        # Setup observation and action spaces
        self._setup_spaces()
        
        # Rendering
        self.window = None
        self.clock = None
        self.window_width = 800
        self.window_height = 600
        
    def _setup_spaces(self):
        """Setup observation and action spaces for all agents"""
        
        # Observation space: combines elements from vertical and horizontal CR envs
        # Own state: [altitude, vz, target_altitude, runway_distance, heading, airspeed]
        # Other aircraft: [relative positions, velocities, conflict flags] for nearby aircraft
        # Communication: [message_type, message_values] if enabled
        
        max_nearby_aircraft = min(self.num_aircraft - 1, 5)  # Observe max 5 other aircraft
        
        obs_dict = {
            # Own aircraft state (6 components)
            "altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
            "vz": spaces.Box(-np.inf, np.inf, dtype=np.float64),
            "target_altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
            "runway_distance": spaces.Box(-np.inf, np.inf, dtype=np.float64),
            "heading": spaces.Box(0, 360, dtype=np.float64),
            "airspeed": spaces.Box(0, 1000, dtype=np.float64),
            
            # Nearby aircraft (relative information)
            "nearby_distances": spaces.Box(0, np.inf, shape=(max_nearby_aircraft,), dtype=np.float64),
            "nearby_bearings": spaces.Box(-180, 180, shape=(max_nearby_aircraft,), dtype=np.float64),
            "nearby_alt_diff": spaces.Box(-np.inf, np.inf, shape=(max_nearby_aircraft,), dtype=np.float64),
            "nearby_conflict_flags": spaces.Box(0, 1, shape=(max_nearby_aircraft,), dtype=np.float64),
            "nearby_rel_velocities": spaces.Box(-np.inf, np.inf, shape=(max_nearby_aircraft, 3), dtype=np.float64),
        }
        
        # Add communication if enabled
        if self.enable_communication:
            obs_dict["received_messages"] = spaces.Box(-1, 1, shape=(3, 4), dtype=np.float64)
        
        self.observation_space = spaces.Dict(obs_dict)
        
        # Action space: [vertical_speed_change, heading_change, speed_change] + optional communication
        action_components = 3
        action_low = [-1.0, -1.0, -1.0]  # Normalized actions
        action_high = [1.0, 1.0, 1.0]
        
        if self.enable_communication:
            action_components += 3  # [message_type, value1, value2]
            action_low += [0, -1.0, -1.0]
            action_high += [10, 1.0, 1.0]
        
        self.action_space = spaces.Box(
            low=np.array(action_low), 
            high=np.array(action_high), 
            dtype=np.float64
        )
        
        # Apply to all agents
        self.observation_spaces = {agent: self.observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space for agent in self.possible_agents}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset BlueSky
        bs.traf.reset()
        
        # Reset environment state
        self.current_step = 0
        self.agents = self.possible_agents.copy()
        self.aircraft_data = {}
        self.conflicts = []
        self.total_intrusions = 0
        self.episode_rewards = {agent: 0.0 for agent in self.agents}
        
        # Initialize aircraft
        self._initialize_aircraft()
        
        # Generate initial conflicts (intruders)
        self._generate_conflicts()
        
        observations = self._get_observations()
        infos = self._get_infos()
        
        return observations, infos
    
    def _initialize_aircraft(self):
        """Initialize aircraft positions and targets"""
        for i, agent_id in enumerate(self.agents):
            # Random initial position
            lat = np.random.uniform(-10, 10)
            lon = np.random.uniform(-10, 10)
            alt = np.random.uniform(ALT_MIN, ALT_MAX)
            
            # Random target altitude
            target_alt = alt + np.random.uniform(-TARGET_ALT_DIF, TARGET_ALT_DIF)
            
            # Random heading and speed
            heading = np.random.uniform(0, 360)
            speed = AC_SPD + np.random.uniform(-20, 20)
            
            # Create aircraft in BlueSky
            acid = f"AC{i:02d}"
            bs.traf.cre(acid, actype="A320", aclat=lat, aclon=lon, achdg=heading, 
                       acspd=speed, acalt=alt)
            bs.traf.swvnav[i] = False
            
            # Store aircraft data
            self.aircraft_data[agent_id] = {
                'acid': acid,
                'index': i,
                'target_altitude': target_alt,
                'initial_pos': [lat, lon, alt],
                'conflicts': set(),
                'total_reward': 0.0,
                'messages': [],
                'last_action': None
            }
    
    def _generate_conflicts(self):
        """Generate potential conflict situations between aircraft"""
        # This creates scenarios where aircraft might conflict
        # Based on your vertical_cr_env._generate_conflicts method
        
        for i, agent_id in enumerate(self.agents):
            aircraft_data = self.aircraft_data[agent_id]
            target_idx = aircraft_data['index']
            
            # Add some controlled conflict scenarios
            for j, other_agent in enumerate(self.agents):
                if i != j:
                    other_data = self.aircraft_data[other_agent]
                    other_idx = other_data['index']
                    
                    # Check if aircraft are close enough to potentially conflict
                    lat_diff = abs(bs.traf.lat[target_idx] - bs.traf.lat[other_idx])
                    lon_diff = abs(bs.traf.lon[target_idx] - bs.traf.lon[other_idx])
                    
                    if lat_diff < 5 and lon_diff < 5:  # Within 5 degrees
                        # Potential conflict situation
                        pass
    
    def step(self, actions: Dict[str, np.ndarray]):
        """Execute one step of the environment"""
        self.current_step += 1
        
        # Apply actions for all agents
        for agent_id in self.agents:
            if agent_id in actions:
                self._apply_action(agent_id, actions[agent_id])
        
        # Update BlueSky simulation
        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()
        
        # Detect conflicts
        self._detect_conflicts()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Check termination conditions
        terminations = self._get_terminations()
        truncations = self._get_truncations()
        
        # Generate observations and info
        observations = self._get_observations()
        infos = self._get_infos()
        
        # Update episode rewards
        for agent_id, reward in rewards.items():
            self.episode_rewards[agent_id] += reward
        
        return observations, rewards, terminations, truncations, infos
    
    def _apply_action(self, agent_id: str, action: np.ndarray):
        """Apply action for a specific agent"""
        aircraft_data = self.aircraft_data[agent_id]
        idx = aircraft_data['index']
        
        # Extract actions
        vertical_speed_change = action[0] * ACTION_2_MS  # Convert to m/s
        heading_change = action[1] * D_HEADING  # Convert to degrees
        speed_change = action[2] * D_SPEED  # Convert to knots
        
        # Apply vertical speed change (from vertical_cr_env)
        if vertical_speed_change >= 0:
            bs.traf.selalt[idx] = 1000000  # High target altitude for climb
            bs.traf.selvs[idx] = vertical_speed_change
        else:
            bs.traf.selalt[idx] = 0  # Low target altitude for descent
            bs.traf.selvs[idx] = vertical_speed_change
        
        # Apply heading change (from horizontal_cr_env)
        new_heading = fn.bound_angle_positive_negative_180(bs.traf.hdg[idx] + heading_change)
        bs.stack.stack(f"HDG {aircraft_data['acid']} {new_heading}")
        
        # Apply speed change
        new_speed = bs.traf.cas[idx] + speed_change
        new_speed = np.clip(new_speed, 100, 500)  # Reasonable speed limits
        bs.stack.stack(f"SPD {aircraft_data['acid']} {new_speed}")
        
        # Store last action
        aircraft_data['last_action'] = action[:3]
        
        # Handle communication if enabled
        if self.enable_communication and len(action) >= 6:
            message_type = int(action[3])
            message_values = action[4:6]
            if message_type > 0:
                self._send_message(agent_id, message_type, message_values)
    
    def _detect_conflicts(self):
        """Detect conflicts between aircraft"""
        current_conflicts = []
        
        # Clear previous conflicts
        for agent_data in self.aircraft_data.values():
            agent_data['conflicts'].clear()
        
        # Check all pairs of aircraft
        agents_list = list(self.agents)
        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                agent1_id = agents_list[i]
                agent2_id = agents_list[j]
                
                idx1 = self.aircraft_data[agent1_id]['index']
                idx2 = self.aircraft_data[agent2_id]['index']
                
                # Calculate lateral distance
                _, lateral_dist = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[idx1], bs.traf.lon[idx1],
                    bs.traf.lat[idx2], bs.traf.lon[idx2]
                )
                
                # Calculate vertical separation
                vertical_sep = abs(bs.traf.alt[idx1] - bs.traf.alt[idx2])
                
                # Check if conflict exists
                if (lateral_dist < INTRUSION_DISTANCE and 
                    vertical_sep < VERTICAL_MARGIN):
                    
                    current_conflicts.append((agent1_id, agent2_id, lateral_dist, vertical_sep))
                    self.aircraft_data[agent1_id]['conflicts'].add(agent2_id)
                    self.aircraft_data[agent2_id]['conflicts'].add(agent1_id)
        
        self.conflicts = current_conflicts
        self.total_intrusions += len(current_conflicts)
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all agents"""
        rewards = {}
        
        for agent_id in self.agents:
            reward = 0.0
            aircraft_data = self.aircraft_data[agent_id]
            idx = aircraft_data['index']
            
            # Conflict penalty (from vertical_cr_env)
            num_conflicts = len(aircraft_data['conflicts'])
            reward += num_conflicts * INTRUSION_PENALTY
            
            # Altitude tracking reward (from vertical_cr_env)
            current_alt = bs.traf.alt[idx]
            target_alt = aircraft_data['target_altitude']
            alt_error = abs(target_alt - current_alt)
            reward += alt_error * ALT_DIF_REWARD_SCALE
            
            # Efficiency penalty for large actions
            if aircraft_data['last_action'] is not None:
                action_magnitude = np.sum(np.abs(aircraft_data['last_action']))
                reward += action_magnitude * (-0.01)  # Small penalty for large actions
            
            # Runway approach reward (simplified)
            runway_distance = bs.tools.geo.kwikdist(RWY_LAT, RWY_LON, 
                                                   bs.traf.lat[idx], bs.traf.lon[idx]) * NM2KM
            
            if runway_distance < DEFAULT_RWY_DIS:
                # Reward for being close to runway
                reward += 0.1
                
                # Check if at correct altitude for approach
                approach_alt_error = abs(current_alt - 1000)  # Target approach altitude
                reward += approach_alt_error * RWY_ALT_DIF_REWARD_SCALE
            
            # Cooperative reward component
            if self.cooperative_reward:
                # Shared penalty for system-wide conflicts
                global_conflict_penalty = -len(self.conflicts) * 2.0
                reward += global_conflict_penalty / len(self.agents)
            
            # Survival bonus
            reward += 0.1  # Small positive reward for staying airborne
            
            rewards[agent_id] = reward
        
        return rewards
    
    def _get_observations(self) -> Dict[str, Dict]:
        """Generate observations for all agents"""
        observations = {}
        
        for agent_id in self.agents:
            obs = self._get_agent_observation(agent_id)
            observations[agent_id] = obs
        
        return observations
    
    def _get_agent_observation(self, agent_id: str) -> Dict:
        """Generate observation for a specific agent"""
        aircraft_data = self.aircraft_data[agent_id]
        idx = aircraft_data['index']
        
        # Own aircraft state
        altitude = bs.traf.alt[idx]
        vz = bs.traf.vs[idx]
        target_altitude = aircraft_data['target_altitude']
        heading = bs.traf.hdg[idx]
        airspeed = bs.traf.tas[idx]
        
        # Runway distance
        runway_distance = bs.tools.geo.kwikdist(RWY_LAT, RWY_LON, 
                                               bs.traf.lat[idx], bs.traf.lon[idx]) * NM2KM
        
        # Normalize own state
        obs = {
            "altitude": np.array([(altitude - ALT_MEAN) / ALT_STD]),
            "vz": np.array([(vz - VZ_MEAN) / VZ_STD]),
            "target_altitude": np.array([(target_altitude - ALT_MEAN) / ALT_STD]),
            "runway_distance": np.array([(runway_distance - RWY_DIS_MEAN) / RWY_DIS_STD]),
            "heading": np.array([heading / 360.0]),
            "airspeed": np.array([airspeed / 500.0])
        }
        
        # Find nearby aircraft
        nearby_info = self._get_nearby_aircraft_info(agent_id)
        obs.update(nearby_info)
        
        # Add communication messages if enabled
        if self.enable_communication:
            messages = aircraft_data['messages'][-3:]  # Last 3 messages
            message_array = np.zeros((3, 4))
            for i, msg in enumerate(messages):
                if i < 3:
                    message_array[i] = [msg['type'], msg['value1'], msg['value2'], msg['urgency']]
            obs["received_messages"] = message_array
        
        return obs
    
    def _get_nearby_aircraft_info(self, agent_id: str) -> Dict:
        """Get information about nearby aircraft"""
        aircraft_data = self.aircraft_data[agent_id]
        idx = aircraft_data['index']
        max_nearby = 5
        
        # Initialize arrays
        distances = np.full(max_nearby, 1000.0)  # Large default distance
        bearings = np.zeros(max_nearby)
        alt_diffs = np.zeros(max_nearby)
        conflict_flags = np.zeros(max_nearby)
        rel_velocities = np.zeros((max_nearby, 3))
        
        # Find nearby aircraft
        nearby_aircraft = []
        for other_agent_id in self.agents:
            if other_agent_id != agent_id:
                other_idx = self.aircraft_data[other_agent_id]['index']
                
                # Calculate distance
                bearing, distance = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    bs.traf.lat[other_idx], bs.traf.lon[other_idx]
                )
                
                nearby_aircraft.append((other_agent_id, distance, bearing, other_idx))
        
        # Sort by distance and take closest
        nearby_aircraft.sort(key=lambda x: x[1])
        
        for i, (other_agent_id, distance, bearing, other_idx) in enumerate(nearby_aircraft[:max_nearby]):
            distances[i] = distance * NM2KM / 100.0  # Normalize
            bearings[i] = bearing / 180.0  # Normalize to [-1, 1]
            alt_diffs[i] = (bs.traf.alt[other_idx] - bs.traf.alt[idx]) / ALT_STD
            conflict_flags[i] = 1.0 if other_agent_id in aircraft_data['conflicts'] else 0.0
            
            # Relative velocities
            rel_velocities[i, 0] = (bs.traf.hdg[other_idx] - bs.traf.hdg[idx]) / 360.0
            rel_velocities[i, 1] = (bs.traf.tas[other_idx] - bs.traf.tas[idx]) / 500.0
            rel_velocities[i, 2] = (bs.traf.vs[other_idx] - bs.traf.vs[idx]) / VZ_STD
        
        return {
            "nearby_distances": distances,
            "nearby_bearings": bearings,
            "nearby_alt_diff": alt_diffs,
            "nearby_conflict_flags": conflict_flags,
            "nearby_rel_velocities": rel_velocities
        }
    
    def _send_message(self, sender_id: str, message_type: int, values: np.ndarray):
        """Send message to nearby agents (if communication enabled)"""
        if not self.enable_communication:
            return
        
        sender_idx = self.aircraft_data[sender_id]['index']
        comm_range = INTRUSION_DISTANCE * 2  # Double the intrusion distance
        
        message = {
            'sender': sender_id,
            'type': message_type,
            'value1': values[0],
            'value2': values[1],
            'urgency': 1.0 if len(self.aircraft_data[sender_id]['conflicts']) > 0 else 0.5
        }
        
        # Send to nearby aircraft
        for receiver_id in self.agents:
            if receiver_id != sender_id:
                receiver_idx = self.aircraft_data[receiver_id]['index']
                
                _, distance = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[sender_idx], bs.traf.lon[sender_idx],
                    bs.traf.lat[receiver_idx], bs.traf.lon[receiver_idx]
                )
                
                if distance <= comm_range:
                    self.aircraft_data[receiver_id]['messages'].append(message)
                    
                    # Keep only recent messages
                    if len(self.aircraft_data[receiver_id]['messages']) > 10:
                        self.aircraft_data[receiver_id]['messages'] = \
                            self.aircraft_data[receiver_id]['messages'][-10:]
    
    def _get_terminations(self) -> Dict[str, bool]:
        """Check if any agents have terminated"""
        terminations = {}
        
        for agent_id in self.agents:
            aircraft_data = self.aircraft_data[agent_id]
            idx = aircraft_data['index']
            
            # Terminate if crashed (altitude too low)
            crashed = bs.traf.alt[idx] <= 0
            
            # Terminate if reached destination (simplified)
            runway_distance = bs.tools.geo.kwikdist(RWY_LAT, RWY_LON, 
                                                   bs.traf.lat[idx], bs.traf.lon[idx]) * NM2KM
            reached_destination = (runway_distance < 2.0 and bs.traf.alt[idx] < 500)
            
            terminations[agent_id] = crashed or reached_destination
        
        return terminations
    
    def _get_truncations(self) -> Dict[str, bool]:
        """Check if episode should be truncated"""
        truncated = self.current_step >= self.max_episode_steps
        return {agent_id: truncated for agent_id in self.agents}
    
    def _get_infos(self) -> Dict[str, Dict]:
        """Generate info dictionaries for all agents"""
        infos = {}
        
        for agent_id in self.agents:
            aircraft_data = self.aircraft_data[agent_id]
            idx = aircraft_data['index']
            
            infos[agent_id] = {
                'conflicts': list(aircraft_data['conflicts']),
                'num_conflicts': len(aircraft_data['conflicts']),
                'total_reward': self.episode_rewards[agent_id],
                'altitude': bs.traf.alt[idx],
                'target_altitude': aircraft_data['target_altitude'],
                'step': self.current_step,
                'runway_distance': bs.tools.geo.kwikdist(RWY_LAT, RWY_LON, 
                                                        bs.traf.lat[idx], bs.traf.lon[idx]) * NM2KM
            }
        
        # Add global information to first agent
        if self.agents:
            first_agent = self.agents[0]
            infos[first_agent]['global_conflicts'] = len(self.conflicts)
            infos[first_agent]['total_intrusions'] = self.total_intrusions
        
        return infos
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        """Render a frame of the environment"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("MARL Conflict Resolution")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((135, 206, 235))  # Sky blue background
        
        # Draw aircraft
        for agent_id in self.agents:
            aircraft_data = self.aircraft_data[agent_id]
            idx = aircraft_data['index']
            
            # Convert lat/lon to screen coordinates (simplified)
            x = int((bs.traf.lon[idx] + 20) * self.window_width / 40)
            y = int((20 - bs.traf.lat[idx]) * self.window_height / 40)
            
            # Choose color based on conflicts
            if len(aircraft_data['conflicts']) > 0:
                color = (255, 0, 0)  # Red for conflicted aircraft
            else:
                color = (0, 255, 0)  # Green for safe aircraft
            
            # Draw aircraft as circle
            pygame.draw.circle(canvas, color, (x, y), 8)
            
            # Draw heading indicator
            heading_rad = np.radians(bs.traf.hdg[idx])
            end_x = x + 20 * np.sin(heading_rad)
            end_y = y - 20 * np.cos(heading_rad)
            pygame.draw.line(canvas, color, (x, y), (end_x, end_y), 2)
            
            # Draw aircraft ID
            font = pygame.font.Font(None, 24)
            text = font.render(f"AC{idx}", True, (0, 0, 0))
            canvas.blit(text, (x + 10, y - 10))
        
        # Draw runway
        runway_x = int((RWY_LON + 20) * self.window_width / 40)
        runway_y = int((20 - RWY_LAT) * self.window_height / 40)
        pygame.draw.rect(canvas, (100, 100, 100), (runway_x - 20, runway_y - 5, 40, 10))
        
        # Display conflict information
        font = pygame.font.Font(None, 36)
        conflict_text = font.render(f"Conflicts: {len(self.conflicts)}", True, (255, 255, 255))
        canvas.blit(conflict_text, (10, 10))
        
        step_text = font.render(f"Step: {self.current_step}", True, (255, 255, 255))
        canvas.blit(step_text, (10, 50))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))
    
    def close(self):
        """Close the environment"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# Wrapper for easier integration with existing training pipelines
def env(**kwargs):
    """Create environment for PettingZoo compatibility"""
    env = MARLConflictResolutionEnv(**kwargs)
    # Add standard PettingZoo wrappers
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
pass
