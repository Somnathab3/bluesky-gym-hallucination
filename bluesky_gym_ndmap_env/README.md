## bluesky_gym_ndmap_env/README.md

# ND-Map Conflict-Resolution Environment

## Overview

This module implements a high-performance conflict detection and resolution environment using the N-Dimensional Map (ND-Map) algorithm from Kuenz's 2015 dissertation. The ND-Map provides efficient 4D spatial-temporal indexing for air traffic conflict detection in large-scale airspace scenarios.

**Important Note**: No public pip package exists for DLR's ND-Map implementation. This module contains a research implementation based on:

> Kuenz, A. (2015). "High Performance Conflict Detection and Resolution for Multi-Dimensional Objects." Dissertation, Deutsches Zentrum für Luft- und Raumfahrt (DLR).

## Key Features

### 4D Hyper-Bisection Index
- **Dimensions**: Latitude, longitude, altitude, time
- **Geodetic Correction**: Cell sizing adjusted by cos φ for accurate distance calculations
- **Dynamic Tiling**: Recursive bisection only where trajectories/zones overlap
- **Tile Arity**: Up to 16 subdivisions per dimension for optimal performance

### Separation Standards
- **Lateral Separation**: 5 NM minimum (configurable)
- **Vertical Separation**: 1,000 ft minimum (configurable)  
- **Temporal Lookahead**: User-configurable time horizon

### Performance Optimizations
- **Broad-Phase Detection**: ND-Map spatial queries for candidate pairs
- **Narrow-Phase Verification**: Analytic closest-point-of-approach (CPA) calculations
- **Dynamic Updates**: Efficient trajectory insertion/modification without full rebuild

## Research Applications

- **Scalability Testing**: Large-scale multi-aircraft conflict scenarios
- **Algorithm Benchmarking**: Compare ND-Map vs traditional pairwise detection
- **Real-Time Performance**: Evaluate computational efficiency under varying traffic densities
- **Hallucination Impact**: Test ML model responses to ND-Map detected vs phantom conflicts

## Usage Example

```python
import gymnasium as gym
import bluesky_gym_ndmap_env

# Create ND-Map environment with custom parameters
env = gym.make("NDMapConflictEnv-v0", 
               airspace_bounds=(45.0, 55.0, 0.0, 10.0),  # European airspace sector
               max_intruders=20,
               tile_arity=16)

obs, info = env.reset()
print(f"ND-Map initialized with {info['ndmap_stats']['total_tiles']} tiles")

# Training loop with conflict detection analytics
for step in range(1000):
    action = your_conflict_resolution_model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Analyze ND-Map performance
    if info["conflicts"]:
        print(f"Step {step}: {len(info['conflicts'])} conflicts detected")
        print(f"ND-Map query time: {info['ndmap_stats']['query_time_ms']:.2f} ms")
    
    if terminated or truncated:
        break
```

## Environment Details

### Observation Space
- **k-nearest intruders**: Relative state vectors [Δlat, Δlon, Δalt, heading, speed]
- **Own-ship state**: Current position, velocity, heading
- **Conflict indicators**: Binary flags for detected conflicts

### Action Space  
- **Heading changes**: ±30° maximum deviation
- **Altitude changes**: ±2000 ft/min climb/descent rates
- **Speed adjustments**: ±10% of current airspeed

### Reward Structure
- **Conflict penalty**: -1.0 per detected conflict
- **Efficiency bonus**: +0.1 for maintaining original route
- **Separation reward**: +0.5 for maintaining safe distances

### Info Dictionary
- **conflicts**: List of conflict details with CPA times and separation distances
- **ndmap_stats**: Performance metrics (total tiles, max depth, query times)

## Implementation Notes

This research implementation follows Chapter 3 of Kuenz's dissertation for algorithmic accuracy while optimizing for Python/NumPy performance. The ND-Map algorithm provides significant computational advantages over O(n²) pairwise conflict detection for scenarios with >10 aircraft.

## Citation

```bibtex
@phdthesis{kuenz2015high,
    title={High Performance Conflict Detection and Resolution for Multi-Dimensional Objects},
    author={Kuenz, Alexander},
    year={2015},
    school={Deutsches Zentrum für Luft- und Raumfahrt (DLR)}
}
```

