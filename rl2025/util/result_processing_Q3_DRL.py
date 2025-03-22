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

##EVERYTHING BELOW HERE IS FOR QUESTION 3##
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rl2025.util.result_processing import Run, rank_runs
from rl2025.exercise3.train_discreterl import train, MOUNTAINCAR_CONFIG
from rl2025.constants import EX3_DISCRETERL_MOUNTAINCAR_CONSTANTS

# Number of seeds to run
N_SEEDS = 10

# Create Run objects for each learning rate configuration
discrete_run_0_02 = Run({"alpha": 0.02, "save_filename": None})
discrete_run_0_02.run_name = "DiscreteRL (alpha=0.02)"

discrete_run_0_002 = Run({"alpha": 0.002, "save_filename": None})
discrete_run_0_002.run_name = "DiscreteRL (alpha=0.002)"

discrete_run_0_0002 = Run({"alpha": 0.0002, "save_filename": None})
discrete_run_0_0002.run_name = "DiscreteRL (alpha=0.0002)"

# File to save the results
SWEEP_RESULTS_FILE = "DiscreteRL-MountainCar-alpha-sweep-results.pkl"

def run_learning_rate_experiment():
    """Run experiments for different learning rates in DiscreteRL Mountain Car"""
    
    # Run experiments for each learning rate
    for seed in range(N_SEEDS):
        print(f"\n===== Seed {seed+1}/{N_SEEDS} =====")
        
        # Configuration 1 (alpha = 0.02)
        print(f"Running DiscreteRL with alpha=0.02")
        config1 = MOUNTAINCAR_CONFIG.copy()
        config1.update({
            "alpha": 0.02,
        })
        config1.update(EX3_DISCRETERL_MOUNTAINCAR_CONSTANTS)
        
        env = gym.make(config1["env"])
        eval_returns, eval_timesteps, times, run_data = train(env, config1, output=False)
        discrete_run_0_02.update(eval_returns, eval_timesteps, times, run_data)
        env.close()
        
        # Configuration 2 (alpha = 0.002)
        print(f"Running DiscreteRL with alpha=0.002")
        config2 = MOUNTAINCAR_CONFIG.copy()
        config2.update({
            "alpha": 0.002,
        })
        config2.update(EX3_DISCRETERL_MOUNTAINCAR_CONSTANTS)
        
        env = gym.make(config2["env"])
        eval_returns, eval_timesteps, times, run_data = train(env, config2, output=False)
        discrete_run_0_002.update(eval_returns, eval_timesteps, times, run_data)
        env.close()
        
        # Configuration 3 (alpha = 0.0002)
        print(f"Running DiscreteRL with alpha=0.0002")
        config3 = MOUNTAINCAR_CONFIG.copy()
        config3.update({
            "alpha": 0.0002,
        })
        config3.update(EX3_DISCRETERL_MOUNTAINCAR_CONSTANTS)
        
        env = gym.make(config3["env"])
        eval_returns, eval_timesteps, times, run_data = train(env, config3, output=False)
        discrete_run_0_0002.update(eval_returns, eval_timesteps, times, run_data)
        env.close()

    # Collect all runs
    all_runs = [discrete_run_0_02, discrete_run_0_002, discrete_run_0_0002]
    
    # Save results
    print(f"\nSaving results to {SWEEP_RESULTS_FILE}")
    with open(SWEEP_RESULTS_FILE, 'wb') as f:
        pickle.dump(all_runs, f)
    
    return all_runs

def analyze_results(runs):
    """Analyze and visualize the results"""
    
    # Print performance statistics
    print("\n=== Performance Statistics ===")
    for run in runs:
        print(f"{run.run_name}:")
        print(f"  Mean final return: {run.final_return_mean:.4f} ± {run.final_return_ste:.4f}")

    # Rank algorithms
    ranked_runs = rank_runs(runs)
    print("\n=== Configurations Ranked by Final Return ===")
    for i, run in enumerate(ranked_runs):
        print(f"{i+1}. {run.run_name}: {run.final_return_mean:.4f}")
    
    # The best learning rate is the one with the highest mean return
    best_run = ranked_runs[0]
    best_learning_rate = float(best_run.run_name.split('=')[1].rstrip(')'))
    print(f"\nBest learning rate: {best_learning_rate}")

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red']
    
    for i, run in enumerate(runs):
        timesteps = run.all_eval_timesteps[0]  # All seeds should have same evaluation timesteps
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
    plt.title('Learning Curves for Different Learning Rates (DiscreteRL)')
    plt.legend()
    plt.grid(True)
    plt.savefig('discreterl_learning_rate_comparison.png')
    plt.show()
    
    return best_learning_rate

if __name__ == "__main__":
    # Either run the experiment or load previous results
    import os
    if os.path.exists(SWEEP_RESULTS_FILE):
        print(f"Loading existing results from {SWEEP_RESULTS_FILE}")
        with open(SWEEP_RESULTS_FILE, 'rb') as f:
            all_runs = pickle.load(f)
    else:
        all_runs = run_learning_rate_experiment()
    
    # Analyze the results
    best_learning_rate = analyze_results(all_runs)
    
    # Print the answer for question3_1
    print("\nAnswer for question3_1:")
    print(f"The learning rate that achieves the highest mean return is: {best_learning_rate}")