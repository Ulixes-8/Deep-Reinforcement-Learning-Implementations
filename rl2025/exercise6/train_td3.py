import os
import copy
import pickle
from collections import defaultdict
import highway_env
import gymnasium as gym
from gymnasium import Space
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from rl2025.constants import EX4_RACETRACK_CONSTANTS as RACETRACK_CONSTANTS
from rl2025.exercise3.replay import ReplayBuffer
from rl2025.util.hparam_sweeping import generate_hparam_configs
from rl2025.util.result_processing import Run

# Import the agents (adjust the import path if needed)
from agents import TD3Ensemble

# Global settings
RENDER = False  # Set to True to visualize environment during evaluation
RESULTS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration for TD3 with Ensemble Critics and Dynamic Policy Delay
TD3_ENSEMBLE_CONFIG = {
    "critic_hidden_size": [32, 32, 32],
    "policy_hidden_size": [32, 32, 32],
    "policy_delay": 2,      # Initial policy delay (will adapt during training)
    "target_noise": 0.2,    # Std of Gaussian noise added to target policy
    "noise_clip": 0.5,      # Clipping for target policy noise
    "num_critics": 4,       # Number of critics in the ensemble
}
TD3_ENSEMBLE_CONFIG.update(RACETRACK_CONSTANTS)
TD3_ENSEMBLE_CONFIG["save_filename"] = os.path.join(RESULTS_DIR, "td3_ensemble_latest.pt")
TD3_ENSEMBLE_CONFIG["algo"] = "TD3Ensemble"


def play_episode(
        env,
        agent,
        replay_buffer=None,
        train=True,
        explore=True,
        render=False,
        max_steps=200,
        batch_size=64,
):
    """Play one episode with the agent in the environment
    
    Args:
        env: gym environment
        agent: RL agent
        replay_buffer: buffer for experience replay
        train: whether to train the agent
        explore: whether to use exploration
        render: whether to render the environment
        max_steps: maximum steps per episode
        batch_size: batch size for updates
        
    Returns:
        episode_timesteps: number of timesteps in the episode
        episode_return: total reward for the episode
        ep_data: dictionary with episode data
    """
    ep_data = defaultdict(list)
    obs, _ = env.reset()
    obs = obs.ravel()
    done = False
    if render:
        env.render()

    episode_timesteps = 0
    episode_return = 0

    while not done:
        action = agent.act(obs, explore=explore)
        nobs, reward, terminated, truncated, _ = env.step(action)
        nobs = nobs.ravel()
        done = terminated or truncated
        
        if train and replay_buffer is not None:
            # Store transition in replay buffer
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            
            # Update the agent if we have enough samples
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                update_info = agent.update(batch)
                
                # Store update information
                for k, v in update_info.items():
                    ep_data[k].append(v)

        episode_timesteps += 1
        episode_return += reward

        if render:
            env.render()

        obs = nobs

    return episode_timesteps, episode_return, ep_data


def train(env: gym.Env, env_eval: gym.Env, config: Dict, output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Execute training of TD3Ensemble on given environment using the provided configuration

    Args:
        env: environment to train on
        env_eval: environment for evaluation
        config: configuration dictionary mapping configuration keys to values
        output: flag whether evaluation results should be printed
        
    Returns:
        eval_returns_all: evaluation returns during training
        eval_timesteps_all: timesteps at which evaluations were performed
        eval_times_all: wall-clock times at which evaluations were performed
        run_data: dictionary with additional training data
    """
    # Reset environment and create observation space
    obs, _ = env.reset()
    obs = obs.ravel()
    observation_space = Space((obs.shape[0],))

    # Create agent
    agent = TD3Ensemble(
        action_space=env.action_space,
        observation_space=observation_space,
        **config
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    # Initialize tracking variables
    timesteps_elapsed = 0
    eval_returns_all = []
    eval_timesteps_all = []
    eval_times_all = []
    run_data = defaultdict(list)
    
    # For tracking dynamic policy delay
    policy_delays = []
    policy_delay_timesteps = []

    # Start training
    start_time = time.time()
    with tqdm(total=config["max_timesteps"]) as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            # Check if we've exceeded the maximum time
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break

            # Schedule hyperparameters
            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
            
            # Track current policy delay
            policy_delays.append(agent.policy_delay)
            policy_delay_timesteps.append(timesteps_elapsed)
            
            # Play an episode
            episode_timesteps, ep_return, ep_data = play_episode(
                env,
                agent,
                replay_buffer,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )
            
            # Update timesteps and progress bar
            timesteps_elapsed += episode_timesteps
            pbar.update(episode_timesteps)
            
            # Store episode data
            for k, v in ep_data.items():
                run_data[k].extend(v)
            run_data["train_ep_returns"].append(ep_return)

            # Run evaluation if it's time
            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_returns = 0
                episodic_returns = []
                
                # Evaluate over multiple episodes
                for _ in range(config["eval_episodes"]):
                    _, episode_return, _ = play_episode(
                        env_eval,
                        agent,
                        replay_buffer=None,  # No updates during evaluation
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=config["episode_length"],
                        batch_size=config["batch_size"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]
                    episodic_returns.append(episode_return)
                
                # Print evaluation results
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed}:\n"
                        f"  Mean return: {eval_returns:.2f}\n"
                        f"  Current policy delay: {agent.policy_delay}\n"
                        f"  Returns: {episodic_returns}"
                    )
                
                # Store evaluation results
                eval_returns_all.append(eval_returns)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(time.time() - start_time)
                
                # Check if we've reached the target return
                if eval_returns >= config["target_return"]+400:
                    pbar.write(
                        f"Reached return {eval_returns} >= target return of {config['target_return']}"
                    )
                    break

    # Save the trained agent
    if config["save_filename"]:
        save_path = agent.save(config["save_filename"])
        print(f"Saved model to: {save_path}")
    
    # Add policy delay data to run_data
    run_data["policy_delays"] = policy_delays
    run_data["policy_delay_timesteps"] = policy_delay_timesteps

    return np.array(eval_returns_all), np.array(eval_timesteps_all), np.array(eval_times_all), run_data


def plot_training_results(eval_returns, eval_timesteps, run_data):
    """
    Plot the training results
    
    Args:
        eval_returns: evaluation returns during training
        eval_timesteps: timesteps at which evaluations were performed
        run_data: dictionary with additional training data
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Evaluation returns
    ax1.plot(eval_timesteps, eval_returns, 'b-', linewidth=2)
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Evaluation Return')
    ax1.set_title('TD3 Ensemble Training Progress')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Policy delay
    if "policy_delays" in run_data and "policy_delay_timesteps" in run_data:
        ax2.plot(run_data["policy_delay_timesteps"], run_data["policy_delays"], 'r-', linewidth=2)
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Policy Delay')
        ax2.set_title('Dynamic Policy Delay Adjustment')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Training TD3 with Ensemble Critics and Dynamic Policy Delay")
    print(f"Configuration: {TD3_ENSEMBLE_CONFIG}")
    
    # Create environments
    env = gym.make(TD3_ENSEMBLE_CONFIG["env"])
    env_eval = gym.make(TD3_ENSEMBLE_CONFIG["env"])
    
    # Train the agent
    eval_returns, eval_timesteps, eval_times, run_data = train(
        env, 
        env_eval, 
        TD3_ENSEMBLE_CONFIG, 
        output=True
    )
    
    # Plot training results
    plot_training_results(eval_returns, eval_timesteps, run_data)
    
    # Print final performance
    print(f"Final evaluation return: {eval_returns[-1]:.2f}")
    
    # Close environments
    env.close()
    env_eval.close()