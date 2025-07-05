#!/usr/bin/env python3
"""
Training script for BlueSky MARL environment
Demonstrates multiple training approaches: individual agents, shared policy, and cooperative training
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
import argparse
import time

# Import our MARL environment (this registers it with gym)
import bluesky_gym_marl
from bluesky_gym_marl.utils import (
    make_sb3_env, 
    make_individual_agent_env,
    evaluate_marl_performance,
    MALRLLogger,
    RewardNormalizer
)


def train_individual_agent(env_id: str = "MARLConflict-v0", 
                          agent_id: str = "aircraft_0",
                          algorithm: str = "PPO",
                          total_timesteps: int = 100_000,
                          save_path: str = "models/individual"):
    """Train a single agent while others act randomly"""
    
    print(f"\n=== Training Individual Agent ({agent_id}) with {algorithm} ===")
    
    # Create base environment
    base_env = gym.make(env_id, 
                       cooperative_reward=False,  # Individual learning
                       communication_enabled=False)
    
    # Extract single agent environment
    env = make_individual_agent_env(base_env, agent_id)
    
    # Create algorithm
    if algorithm == "PPO":
        model = PPO("MlpPolicy", env, 
                   learning_rate=3e-4,
                   n_steps=2048,
                   batch_size=64,
                   n_epochs=10,
                   gamma=0.99,
                   gae_lambda=0.95,
                   clip_range=0.2,
                   ent_coef=0.01,
                   verbose=1)
    elif algorithm == "A2C":
        model = A2C("MlpPolicy", env,
                   learning_rate=7e-4,
                   n_steps=5,
                   gamma=0.99,
                   gae_lambda=1.0,
                   ent_coef=0.01,
                   verbose=1)
    elif algorithm == "SAC":
        model = SAC("MlpPolicy", env,
                   learning_rate=3e-4,
                   buffer_size=1000000,
                   learning_starts=100,
                   batch_size=256,
                   tau=0.005,
                   gamma=0.99,
                   verbose=1)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Setup callbacks
    os.makedirs(save_path, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix=f"individual_{algorithm.lower()}_{agent_id}"
    )
    
    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, 
                callback=checkpoint_callback)
    training_time = time.time() - start_time
    
    # Save final model
    model_path = os.path.join(save_path, f"individual_{algorithm.lower()}_{agent_id}_final.zip")
    model.save(model_path)
    
    print(f"Training completed in {training_time:.2f}s")
    print(f"Model saved to: {model_path}")
    
    # Quick evaluation
    print("\n=== Quick Evaluation ===")
    evaluate_individual_agent(env, model, episodes=5)
    
    env.close()
    return model_path


def train_shared_policy(env_id: str = "MARLConflict-Cooperative-v0",
                       algorithm: str = "PPO", 
                       total_timesteps: int = 200_000,
                       num_envs: int = 4,
                       save_path: str = "models/shared"):
    """Train a shared policy across all agents using vectorized environments"""
    
    print(f"\n=== Training Shared Policy with {algorithm} ===")
    
    # Create vectorized environment
    vec_env = make_sb3_env(env_id, 
                          num_envs=num_envs,
                          cooperative_reward=True,
                          communication_enabled=True,
                          sector_based=True)
    
    vec_env = VecMonitor(vec_env)
    
    # Create algorithm with cooperative settings
    if algorithm == "PPO":
        model = PPO("MlpPolicy", vec_env,
                   learning_rate=3e-4,
                   n_steps=1024,  # Shorter for multi-agent
                   batch_size=256,
                   n_epochs=10,
                   gamma=0.99,
                   gae_lambda=0.95,
                   clip_range=0.2,
                   ent_coef=0.02,  # Higher exploration for coordination
                   verbose=1)
    elif algorithm == "A2C":
        model = A2C("MlpPolicy", vec_env,
                   learning_rate=7e-4,
                   n_steps=5,
                   gamma=0.99,
                   gae_lambda=1.0,
                   ent_coef=0.02,
                   verbose=1)
    else:
        raise ValueError(f"Algorithm {algorithm} not supported for shared policy training")
    
    # Setup callbacks
    os.makedirs(save_path, exist_ok=True)
    
    # Evaluation environment (single env for eval)
    eval_env = make_sb3_env(env_id, num_envs=1,
                           cooperative_reward=True,
                           communication_enabled=True,
                           sector_based=True)
    
    eval_callback = EvalCallback(eval_env, 
                                best_model_save_path=save_path,
                                log_path=save_path,
                                eval_freq=10000,
                                deterministic=True,
                                render=False)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=save_path,
        name_prefix=f"shared_{algorithm.lower()}"
    )
    
    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, 
                callback=[eval_callback, checkpoint_callback])
    training_time = time.time() - start_time
    
    # Save final model
    model_path = os.path.join(save_path, f"shared_{algorithm.lower()}_final.zip")
    model.save(model_path)
    
    print(f"Training completed in {training_time:.2f}s")
    print(f"Model saved to: {model_path}")
    
    vec_env.close()
    eval_env.close()
    return model_path


def evaluate_individual_agent(env, model, episodes: int = 10):
    """Evaluate single agent performance"""
    total_rewards = []
    episode_lengths = []
    conflicts_per_episode = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        episode_length = 0
        episode_conflicts = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            episode_length += 1
            
            # Track conflicts if available in info
            if 'num_conflicts' in info:
                episode_conflicts += info['num_conflicts']
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        conflicts_per_episode.append(episode_conflicts)
        
        print(f"Episode {episode + 1}: Reward={total_reward:.2f}, "
              f"Length={episode_length}, Conflicts={episode_conflicts}")
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Mean Conflicts: {np.mean(conflicts_per_episode):.1f} ± {np.std(conflicts_per_episode):.1f}")


def evaluate_shared_policy(env_id: str, model_path: str, episodes: int = 10):
    """Evaluate shared policy in original multi-agent environment"""
    print(f"\n=== Evaluating Shared Policy ===")
    
    # Load the trained model
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Cannot determine algorithm from path: {model_path}")
    
    # Create multi-agent environment for evaluation
    base_env = gym.make(env_id,
                       cooperative_reward=True,
                       communication_enabled=True,
                       sector_based=True)
    
    # Convert to individual agent format for the trained model
    agent_env = make_individual_agent_env(base_env, "aircraft_0")
    
    # Run evaluation
    evaluate_individual_agent(agent_env, model, episodes)
    
    agent_env.close()


def run_experiment_suite():
    """Run a comprehensive experiment comparing different approaches"""
    print("=" * 60)
    print("MARL AIR TRAFFIC CONTROL - EXPERIMENT SUITE")
    print("=" * 60)
    
    results = {}
    
    # 1. Individual agent training
    print("\n" + "="*50)
    print("EXPERIMENT 1: Individual Agent Training")
    print("="*50)
    
    for algorithm in ["PPO", "A2C"]:
        model_path = train_individual_agent(
            env_id="MARLConflict-Simple-v0",
            algorithm=algorithm,
            total_timesteps=50_000,
            save_path=f"models/individual_{algorithm.lower()}"
        )
        results[f"individual_{algorithm}"] = model_path
    
    # 2. Shared policy training
    print("\n" + "="*50)
    print("EXPERIMENT 2: Shared Policy Training")
    print("="*50)
    
    for algorithm in ["PPO", "A2C"]:
        model_path = train_shared_policy(
            env_id="MARLConflict-Cooperative-v0",
            algorithm=algorithm,
            total_timesteps=100_000,
            num_envs=4,
            save_path=f"models/shared_{algorithm.lower()}"
        )
        results[f"shared_{algorithm}"] = model_path
    
    # 3. Comparative evaluation
    print("\n" + "="*50)
    print("EXPERIMENT 3: Comparative Evaluation")
    print("="*50)
    
    for exp_name, model_path in results.items():
        print(f"\nEvaluating {exp_name}:")
        if "shared" in exp_name:
            evaluate_shared_policy("MARLConflict-Cooperative-v0", model_path, episodes=5)
        else:
            # For individual models, create environment and evaluate
            base_env = gym.make("MARLConflict-Simple-v0", cooperative_reward=False)
            agent_env = make_individual_agent_env(base_env, "aircraft_0")
            
            if "ppo" in exp_name:
                model = PPO.load(model_path)
            else:
                model = A2C.load(model_path)
            
            evaluate_individual_agent(agent_env, model, episodes=5)
            agent_env.close()
    
    print("\n" + "="*60)
    print("EXPERIMENT SUITE COMPLETED")
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train MARL Air Traffic Control")
    parser.add_argument("--mode", choices=["individual", "shared", "suite"], 
                       default="individual", help="Training mode")
    parser.add_argument("--algorithm", choices=["PPO", "A2C", "SAC"], 
                       default="PPO", help="RL algorithm")
    parser.add_argument("--env", default="MARLConflict-v0", 
                       help="Environment ID")
    parser.add_argument("--timesteps", type=int, default=100_000,
                       help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--save-path", default="models",
                       help="Path to save models")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    if args.mode == "individual":
        train_individual_agent(
            env_id=args.env,
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            save_path=args.save_path
        )
    elif args.mode == "shared":
        train_shared_policy(
            env_id=args.env,
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            save_path=args.save_path
        )
    elif args.mode == "suite":
        run_experiment_suite()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
