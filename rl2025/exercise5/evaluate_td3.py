import os
import gymnasium as gym
from gymnasium import Space
from typing import List, Dict
import numpy as np

# Import the TD3 agent and its evaluation function
from agents import TD3
from train_td3 import play_episode

# Import constants (assuming these provide keys like "env", "eval_episodes", "episode_length", "batch_size")
from rl2025.constants import EX4_RACETRACK_CONSTANTS as RACETRACK_CONSTANTS

# Global settings and configuration
RENDER = False
RESULTS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

TD3_CONFIG = {
    "critic_hidden_size": [32, 32, 32],
    "policy_hidden_size": [32, 32, 32],
    "policy_delay": 2,        # Update policy every 2 critic updates
    "target_noise": 0.2,      # Std dev of Gaussian noise for target policy
    "noise_clip": 0.5,        # Clipping for target noise
}
TD3_CONFIG.update(RACETRACK_CONSTANTS)
TD3_CONFIG["save_filename"] = os.path.join(RESULTS_DIR, "td3_latest.pt")
TD3_CONFIG["algo"] = "TD3"

def evaluate(env: gym.Env, config: Dict, output: bool = True) -> List[float]:
    """
    Evaluate a pre-trained TD3 agent on the provided environment.

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
    
    # Initialize the TD3 agent
    agent = TD3(
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
            # Note: We pass 0 as the third argument (as in the DDPG evaluation)
            _, episode_return, _ = play_episode(
                env,
                agent,
                0,  # dummy value; adjust if your TD3 play_episode requires a different argument
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

if __name__ == "__main__":
    runs = 10
    final_returns = []
    for i in range(runs):
        env = gym.make(TD3_CONFIG["env"])
        returns = evaluate(env, TD3_CONFIG)
        print(f"Run {i+1}/{runs}: Max return:", np.max(returns))
        final_returns.append(np.max(returns))        
        env.close()

    # mean return over all runs
    print("Final mean return:", np.mean(final_returns))