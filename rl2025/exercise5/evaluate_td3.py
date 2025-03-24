import os
import gymnasium as gym
from gymnasium import Space
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

from agents import TD3Ensemble
from train_td3 import play_episode

from rl2025.constants import EX4_RACETRACK_CONSTANTS as RACETRACK_CONSTANTS

RENDER = False
RESULTS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

TD3_CONFIG = {
    "critic_hidden_size": [32, 32, 32],
    "policy_hidden_size": [32, 32, 32],
    "policy_delay": 2,        # Update policy every 2 critic updates
    "target_noise": 0.2,      # Std dev of Gaussian noise for target policy
    "noise_clip": 0.5,        # Clipping for target noise
    "num_critics": 4,         # Number of critics in the ensemble
}
TD3_CONFIG.update(RACETRACK_CONSTANTS)
TD3_CONFIG["save_filename"] = os.path.join(RESULTS_DIR, "td3_ensemble_latest.pt")
TD3_CONFIG["algo"] = "TD3Ensemble"

def evaluate(env: gym.Env, config: Dict, output: bool = True) -> List[float]:
    """
    Evaluate a pre-trained TD3Ensemble agent on the provided environment.
    :param env: gym environment to evaluate on
    :param config: configuration dictionary with keys like "env", "eval_episodes",
                   "episode_length", "batch_size", and "save_filename"
    :param output: if True, prints evaluation results
    :return: list of average evaluation returns (one per loop)
    """
    # Reset environment and flatten observation
    obs, _ = env.reset()
    obs = obs.ravel()
    observation_space = Space((obs.shape[0],))
    
    # Initialize the TD3Ensemble agent
    agent = TD3Ensemble(
        action_space=env.action_space,
        observation_space=observation_space,
        **config
    )
    
    # Restore the pre-trained model
    try:
        agent.restore(config["save_filename"])
    except Exception as e:
        raise ValueError(f"Could not load model from {config['save_filename']}: {e}")
    
    eval_returns_all = []
    
    # Execute 3 evaluation loops; each loop averages over config["eval_episodes"] episodes
    for loop in range(3):
        eval_returns = 0
        for _ in range(config["eval_episodes"]):
            episode_timesteps, episode_return, _ = play_episode(
                env,
                agent,
                replay_buffer=None,  # No updates during evaluation
                train=False,
                explore=False,
                render=RENDER,
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )
            eval_returns += episode_return / config["eval_episodes"]
        eval_returns_all.append(eval_returns)
    
    if output:
        print("Evaluation returns:", eval_returns_all)
    
    return eval_returns_all

def plot_evaluation_results(returns: List[float]):
    """
    Plot the evaluation results
    
    Args:
        returns: list of returns for each evaluation loop
    """
    plt.figure(figsize=(10, 6))
    
    # Plot individual loop returns
    plt.bar(range(len(returns)), returns, alpha=0.7)
    
    # Plot mean return line
    mean_return = np.mean(returns)
    plt.axhline(y=mean_return, color='r', linestyle='--', 
               label=f'Mean Return: {mean_return:.2f}')
    
    # Add labels and title
    plt.xlabel('Evaluation Loop', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title('TD3Ensemble Evaluation Results', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    runs = 10
    final_returns = []
    for i in range(runs):
        env = gym.make(TD3_CONFIG["env"])
        returns = evaluate(env, TD3_CONFIG)
        print(f"Run {i+1}/{runs}: Max return:", np.max(returns))
        final_returns.append(np.max(returns))
        
        # Plot results for this run
        # plot_evaluation_results(returns)
        
        env.close()
    # mean return over all runs
    print("Final mean return:", np.mean(final_returns))