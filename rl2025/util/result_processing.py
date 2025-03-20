import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class Run:

    def __init__(self, config: Dict):
        self._config = config
        self._run_name = None

        self._final_returns = []
        self._train_times = []
        self._run_data = []
        self._agent_weights_filenames = []

        self._run_ids = []
        self._all_eval_timesteps = []
        self._all_returns = []

    def update(self, eval_returns, eval_timesteps, times=None, run_data=None):

        self._run_ids.append(len(self._run_ids))
        if self._config['save_filename'] is not None:
            self._agent_weights_filenames.append(self._config['save_filename'])
            self._config['save_filename'] = None

        self._all_eval_timesteps.append(eval_timesteps)
        self._all_returns.append(eval_returns)
        self._final_returns.append(eval_returns[-1])
        if times is not None:
            self._train_times.append(times[-1])
        if run_data is not None:
            self._run_data.append(run_data)

    def set_save_filename(self, filename):
        if self._config["save_filename"] is not None:
            print(f"Warning: Save filename already set in config. Overwriting to {filename}.")

        self._config['save_filename'] = f"{filename}.pt"

    @property
    def run_name(self):
        return self._run_name

    @run_name.setter
    def run_name(self, name):
        self._run_name = name

    @property
    def final_return_mean(self) -> float:
        final_returns = np.array(self._final_returns)
        return final_returns.mean()

    @property
    def final_return_ste(self) -> float:
        final_returns = np.array(self._final_returns)
        return np.std(final_returns, ddof=1) / np.sqrt(np.size(final_returns))

    @property
    def final_return_iqm(self) -> float:
        final_returns = np.array(self.final_returns)
        q1 = np.percentile(final_returns, 25)
        q3 = np.percentile(final_returns, 75)
        trimmed_ids = np.nonzero(np.logical_and(final_returns >= q1, final_returns <= q3))
        trimmed_returns = final_returns[trimmed_ids]
        return trimmed_returns.mean()

    @property
    def final_returns(self) -> np.ndarray:
        return np.array(self._final_returns)

    @property
    def train_times(self) -> np.ndarray:
        return np.array(self._train_times)

    @property
    def config(self):
        return self._config

    @property
    def run_ids(self) -> List[int]:
        return self._run_ids

    @property
    def agent_weights_filenames(self) -> List[str]:
        return self._agent_weights_filenames

    @property
    def run_data(self) -> List[Dict]:
        return self._run_data

    @property
    def all_eval_timesteps(self) -> np.ndarray:
        return np.array(self._all_eval_timesteps)

    @property
    def all_returns(self) -> np.ndarray:
        return np.array(self._all_returns)


# The helper functions below are provided to help you process the results of your runs.

def rank_runs(runs: List[Run]):
    """Sorts runs by mean final return, highest to lowest."""

    return sorted(runs, key=lambda x: x.final_return_mean, reverse=True)


def get_best_saved_run(runs:List[Run]) -> Tuple[Run, str]:
    """Returns the run with the highest mean final return and the filename of the saved weights of its highest scoring
    seed, if it exists."""

    ranked_runs = rank_runs(runs)
    best_run = ranked_runs[0]

    if best_run.agent_weights_filenames:
        best_run_id = np.argmax(best_run.final_returns)
        return best_run, best_run.agent_weights_filenames[best_run_id]
    else:
        raise ValueError(f"No saved runs found for highest mean final returns run {best_run.run_name}.")
    

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from rl2025.util.result_processing import Run, rank_runs
from rl2025.exercise2.train_q_learning import train as train_q
from rl2025.exercise2.train_monte_carlo import train as train_mc
from rl2025.constants import EX2_QL_CONSTANTS, EX2_MC_CONSTANTS

# Number of seeds to run
n_seeds = 10

# Create Run objects for each configuration
ql_run1 = Run({"alpha": 0.05, "epsilon": 0.9, "gamma": 0.99, "save_filename": None})
ql_run1.run_name = "QL (γ=0.99)"

ql_run2 = Run({"alpha": 0.05, "epsilon": 0.9, "gamma": 0.8, "save_filename": None})
ql_run2.run_name = "QL (γ=0.8)"

mc_run1 = Run({"epsilon": 0.9, "gamma": 0.99, "save_filename": None})
mc_run1.run_name = "MC (γ=0.99)"

mc_run2 = Run({"epsilon": 0.9, "gamma": 0.8, "save_filename": None})
mc_run2.run_name = "MC (γ=0.8)"


non_slippery = False

# Run Q-Learning experiments
for seed in range(n_seeds):
    print(f"Q-Learning: Seed {seed+1}/{n_seeds}")
    
    # Set random seeds
    # random.seed(seed)
    # np.random.seed(seed)
    
    # Config 1 (gamma = 0.99)
    ql_config1 = {
        "eval_freq": 1000,
        "alpha": 0.05,
        "epsilon": 0.9,
        "gamma": 0.99,
    }
    ql_config1.update(EX2_QL_CONSTANTS)
    
    if non_slippery:
        env = gym.make(ql_config1["env"], is_slippery=False)
        print("Non-slippery")
    else: 
        env = gym.make(ql_config1["env"])
        print("Slippery")

    _, eval_returns, _, _ = train_q(env, ql_config1)
    eval_timesteps = [ql_config1["eval_freq"] * (i+1) for i in range(len(eval_returns))]
    ql_run1.update(eval_returns, eval_timesteps)
    
    # Config 2 (gamma = 0.8)
    ql_config2 = {
        "eval_freq": 1000,
        "alpha": 0.05,
        "epsilon": 0.9,
        "gamma": 0.8,
    }
    ql_config2.update(EX2_QL_CONSTANTS)
    
    if non_slippery:
        env = gym.make(ql_config2["env"], is_slippery=False)
    else:
        env = gym.make(ql_config2["env"])

    _, eval_returns, _, _ = train_q(env, ql_config2)
    eval_timesteps = [ql_config2["eval_freq"] * (i+1) for i in range(len(eval_returns))]
    ql_run2.update(eval_returns, eval_timesteps)

# Run Monte Carlo experiments
for seed in range(n_seeds):
    print(f"Monte Carlo: Seed {seed+1}/{n_seeds}")
    
    # Set random seeds
    # random.seed(seed)
    # np.random.seed(seed)
    
    # Config 1 (gamma = 0.99)
    mc_config1 = {
        "eval_freq": 5000,
        "epsilon": 0.9,
        "gamma": 0.99,
    }
    mc_config1.update(EX2_MC_CONSTANTS)
    
    if non_slippery:
        env = gym.make(mc_config1["env"], is_slippery=False)
        print("Non-slippery")
    else:
        env = gym.make(mc_config1["env"])
        print("Slippery")

    _, eval_returns, _, _ = train_mc(env, mc_config1)
    eval_timesteps = [mc_config1["eval_freq"] * (i+1) for i in range(len(eval_returns))]
    mc_run1.update(eval_returns, eval_timesteps)
    
    # Config 2 (gamma = 0.8)
    mc_config2 = {
        "eval_freq": 5000,
        "epsilon": 0.9,
        "gamma": 0.8,
    }
    mc_config2.update(EX2_MC_CONSTANTS)
    
    if non_slippery:
        env = gym.make(mc_config2["env"], is_slippery=False)
    else:
        env = gym.make(mc_config2["env"])

    _, eval_returns, _, _ = train_mc(env, mc_config2)
    eval_timesteps = [mc_config2["eval_freq"] * (i+1) for i in range(len(eval_returns))]
    mc_run2.update(eval_returns, eval_timesteps)

# Analyze results
all_runs = [ql_run1, ql_run2, mc_run1, mc_run2]

# Print performance statistics
print("\n=== Performance Statistics ===")
for run in all_runs:
    print(f"{run.run_name}:")
    print(f"  Mean final return: {run.final_return_mean:.4f} ± {run.final_return_ste:.4f}")

# Rank algorithms
ranked_runs = rank_runs(all_runs)
print("\n=== Algorithms Ranked by Final Return ===")
for i, run in enumerate(ranked_runs):
    print(f"{i+1}. {run.run_name}: {run.final_return_mean:.4f}")

# Plot learning curves
plt.figure(figsize=(10, 6))
colors = ['blue', 'skyblue', 'red', 'salmon']

for i, run in enumerate(all_runs):
    timesteps = run.all_eval_timesteps[0]
    returns_mean = np.mean(run.all_returns, axis=0)
    returns_std = np.std(run.all_returns, axis=0)
    
    plt.plot(timesteps, returns_mean, color=colors[i], label=run.run_name)
    plt.fill_between(
        timesteps, 
        returns_mean - returns_std, 
        returns_mean + returns_std, 
        color=colors[i], alpha=0.2
    )

plt.xlabel('Training Steps')
plt.ylabel('Mean Return')
plt.title('Learning Curves for Q-Learning and Monte Carlo')
plt.legend()
plt.grid(True)
plt.savefig('rl_comparison.png')
plt.show()