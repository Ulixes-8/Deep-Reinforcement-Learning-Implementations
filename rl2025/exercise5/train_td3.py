import copy
import pickle
from collections import defaultdict
import highway_env
import gymnasium as gym
from gymnasium import Space
import highway_env as hiv
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import os

from rl2025.constants import EX4_RACETRACK_CONSTANTS as RACETRACK_CONSTANTS
from rl2025.exercise3.replay import ReplayBuffer
from rl2025.util.hparam_sweeping import generate_hparam_configs
from rl2025.util.result_processing import Run
from agents import TD3  # Import the TD3 agent

RENDER = False  # FALSE FOR FASTER TRAINING / TRUE TO VISUALIZE ENVIRONMENT DURING EVALUATION

ENV = "RACETRACK"

# Create results directory if it doesn't exist
RESULTS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Base configuration for TD3 inherited from DDPG
TD3_CONFIG = {
    "critic_hidden_size": [32, 32, 32],
    "policy_hidden_size": [32, 32, 32],
    "policy_delay": 2,  # Update policy every 2 critic updates
    "target_noise": 0.2,  # Std of Gaussian noise added to target policy
    "noise_clip": 0.5,  # Clipping for target policy noise
}
TD3_CONFIG.update(RACETRACK_CONSTANTS)
TD3_CONFIG["save_filename"] = os.path.join(RESULTS_DIR, "td3_latest.pt")
TD3_CONFIG["algo"] = "TD3"


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
    """Play one episode with the agent in the environment"""
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
        if train:
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                new_data = agent.update(batch)
                for k, v in new_data.items():
                    ep_data[k].append(v)

        episode_timesteps += 1
        episode_return += reward

        if render:
            env.render()

        obs = nobs

    return episode_timesteps, episode_return, ep_data


def train(env: gym.Env, env_eval: gym.Env, config: Dict, output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Execute training of TD3 on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param env_eval (gym.Env): environment for evaluation
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): eval returns during training, times of evaluation
    """
    # Creating a custom Space
    obs, _ = env.reset()
    obs = obs.ravel()
    observation_space = Space((obs.shape[0],))

    timesteps_elapsed = 0

    agent = TD3(
        action_space=env.action_space, observation_space=observation_space, **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
    eval_timesteps_all = []
    eval_times_all = []
    run_data = defaultdict(list)

    start_time = time.time()

    with tqdm(total=config["max_timesteps"]) as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break

            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
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
            timesteps_elapsed += episode_timesteps
            pbar.update(episode_timesteps)
            for k, v in ep_data.items():
                run_data[k].extend(v)
            run_data["train_ep_returns"].append(ep_return)

            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_returns = 0
                episodic_returns = []
                for _ in range(config["eval_episodes"]):
                    _, episode_return, _ = play_episode(
                        env_eval,
                        agent,
                        replay_buffer,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=config["episode_length"],
                        batch_size=config["batch_size"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]
                    episodic_returns.append(episode_return)
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean returns of {eval_returns}. {episodic_returns}\n"
                    )
                eval_returns_all.append(eval_returns)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(time.time() - start_time)
                if eval_returns >= config["target_return"]:
                    pbar.write(
                        f"Reached return {eval_returns} >= target return of {config['target_return']}"
                    )
                    break

    if config["save_filename"]:
        print("Saving to: ", agent.save(config["save_filename"]))

    return np.array(eval_returns_all), np.array(eval_timesteps_all), np.array(eval_times_all), run_data


if __name__ == "__main__":
    CONFIG = TD3_CONFIG

    env = gym.make(CONFIG["env"])
    env_eval = gym.make(CONFIG["env"])

    _ = train(env, env_eval, CONFIG)

    env.close()
    env_eval.close()