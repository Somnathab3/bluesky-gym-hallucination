````markdown
## BlueSky-Gym
A gymnasium-style library for standardized Reinforcement Learning research in Air Traffic Management, developed in Python.

Built on [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) and The Farama Foundation's [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

<p align="center">
    <img src="https://github.com/user-attachments/assets/6ae83579-78af-4cb7-8096-3a10af54a5c5" width="50%" height="50%"><br/>
    <em>An example trained agent attempting the merge environment available in BlueSky-Gym.</em>
</p>

For a complete list of currently available environments, see [bluesky_gym/envs/README.md](bluesky_gym/envs/README.md).

---

## Changelog (7 July 2025)
- **Installation Fix:** Added instructions to clone the `main_bluesky` branch of BlueSky-Gym for a compatible, local BlueSky-Simulator.  
- **New Custom Environments:** Two enroute control scenarios:
  - **CustomHorizontalCREnv:** Horizontal conflict resolution with built-in hallucination detection.
  - **CustomVerticalCREnv:** Vertical conflict resolution focused on climb/descent conflicts.

---

## Installation

1. **Clone the repository and switch to the `main_bluesky` branch:**  
   ```bash
   git clone https://github.com/TUDelft-CNS-ATM/bluesky-gym.git
   cd bluesky-gym
   git checkout main_bluesky
````

2. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Install BlueSky-Gym:**

   ```bash
   pip install .
   ```

> **Note:** The pip package name is `bluesky-gym`, but import it in code as `bluesky_gym`.

---

## Usage

1. **Register environments:**

   ```python
   import gymnasium as gym
   import bluesky_gym
   bluesky_gym.register_envs()
   ```

2. **Standard example (MergeEnv):**

   ```python
   env = gym.make('MergeEnv-v0', render_mode='human')

   obs, info = env.reset()
   done = truncated = False
   while not (done or truncated):
       action = ...  # Your agent's action
       obs, reward, done, truncated, info = env.step(action)
   ```

3. **Custom enroute control environments:**

   * **Horizontal Conflict Resolution:**

     ```python
     env_horiz = gym.make('CustomHorizontalCREnv-v0')
     ```
   * **Vertical Conflict Resolution:**

     ```python
     env_vert = gym.make('CustomVerticalCREnv-v0')
     ```

4. **Training with Stable Baselines 3:**

   ```python
   from stable_baselines3 import SAC

   model = SAC("MultiInputPolicy", env_horiz)
   model.learn(total_timesteps=1e6)
   model.save('sac_horizontal')
   ```

---

## Contributing & Assistance

Interested in contributing or need help? Join the BlueSky-Gym [Discord](https://discord.gg/s7CdxcSX) or open an issue.
Check out our [roadmap](https://github.com/TUDelft-CNS-ATM/bluesky-gym/issues/24) for ideas and priorities.

---

## Citing

If you use BlueSky-Gym, please cite:

```bibtex
@misc{bluesky-gym,
  author = {Groot, DJ and Leto, G and Vlaskin, A and Moec, A and Ellerbroek, J},
  title = {BlueSky-Gym: Reinforcement Learning Environments for Air Traffic Applications},
  year = {2024},
  journal = {SESAR Innovation Days 2024},
}
```

*Add your publications using `BlueSky-Gym` by submitting a pull request!*

```
```
