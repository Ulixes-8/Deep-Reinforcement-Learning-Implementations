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

##EVERYTHING BELOW HERE IS FOR QUESTION 3 (DQN)##
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rl2025.util.result_processing import Run, rank_runs
from rl2025.exercise3.train_dqn import train, MOUNTAINCAR_CONFIG
from rl2025.constants import EX3_DQN_MOUNTAINCAR_CONSTANTS
from rl2025.exercise3.replay import ReplayBuffer

# Number of seeds to run
N_SEEDS = 10

# Create Run objects for Linear decay strategy with different exploration fractions
linear_run_099 = Run({"epsilon_decay_strategy": "linear", "exploration_fraction": 0.99, "save_filename": None})
linear_run_099.run_name = "Linear (ef=0.99)"

linear_run_075 = Run({"epsilon_decay_strategy": "linear", "exploration_fraction": 0.75, "save_filename": None})
linear_run_075.run_name = "Linear (ef=0.75)"

linear_run_001 = Run({"epsilon_decay_strategy": "linear", "exploration_fraction": 0.01, "save_filename": None})
linear_run_001.run_name = "Linear (ef=0.01)"

# Create Run objects for Exponential decay strategy with different decay factors
exp_run_10 = Run({"epsilon_decay_strategy": "exponential", "epsilon_decay": 1.0, "save_filename": None})
exp_run_10.run_name = "Exponential (decay=1.0)"

exp_run_05 = Run({"epsilon_decay_strategy": "exponential", "epsilon_decay": 0.5, "save_filename": None})
exp_run_05.run_name = "Exponential (decay=0.5)"

exp_run_000001 = Run({"epsilon_decay_strategy": "exponential", "epsilon_decay": 0.00001, "save_filename": None})
exp_run_000001.run_name = "Exponential (decay=0.00001)"

# File to save the results for linear decay
LINEAR_SWEEP_RESULTS_FILE = "DQN-MountainCar-sweep-decay-linear-results.pkl"
# File to save the results for exponential decay
EXP_SWEEP_RESULTS_FILE = "DQN-MountainCar-sweep-decay-exponential-results.pkl"

def run_linear_strategy_experiment():
    """Run experiments for different exploration fractions with linear decay in DQN Mountain Car"""
    
    # Run experiments for each exploration fraction
    for seed in range(N_SEEDS):
        print(f"\n===== Seed {seed+1}/{N_SEEDS} =====")
        
        # Configuration 1 (exploration_fraction = 0.99)
        print(f"Running DQN with linear decay, exploration_fraction=0.99")
        config1 = MOUNTAINCAR_CONFIG.copy()
        config1.update({
            "epsilon_decay_strategy": "linear",
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "exploration_fraction": 0.99,
            "epsilon_decay": None
        })
        config1.update(EX3_DQN_MOUNTAINCAR_CONSTANTS)
        
        env = gym.make(config1["env"])
        eval_returns, eval_timesteps, times, run_data = train(env, config1, output=False)
        linear_run_099.update(eval_returns, eval_timesteps, times, run_data)
        env.close()
        
        # Configuration 2 (exploration_fraction = 0.75)
        print(f"Running DQN with linear decay, exploration_fraction=0.75")
        config2 = MOUNTAINCAR_CONFIG.copy()
        config2.update({
            "epsilon_decay_strategy": "linear",
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "exploration_fraction": 0.75,
            "epsilon_decay": None
        })
        config2.update(EX3_DQN_MOUNTAINCAR_CONSTANTS)
        
        env = gym.make(config2["env"])
        eval_returns, eval_timesteps, times, run_data = train(env, config2, output=False)
        linear_run_075.update(eval_returns, eval_timesteps, times, run_data)
        env.close()
        
        # Configuration 3 (exploration_fraction = 0.01)
        print(f"Running DQN with linear decay, exploration_fraction=0.01")
        config3 = MOUNTAINCAR_CONFIG.copy()
        config3.update({
            "epsilon_decay_strategy": "linear",
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "exploration_fraction": 0.01,
            "epsilon_decay": None
        })
        config3.update(EX3_DQN_MOUNTAINCAR_CONSTANTS)
        
        env = gym.make(config3["env"])
        eval_returns, eval_timesteps, times, run_data = train(env, config3, output=False)
        linear_run_001.update(eval_returns, eval_timesteps, times, run_data)
        env.close()

    # Collect all linear decay runs
    linear_runs = [linear_run_099, linear_run_075, linear_run_001]
    
    # Save results
    print(f"\nSaving linear decay results to {LINEAR_SWEEP_RESULTS_FILE}")
    with open(LINEAR_SWEEP_RESULTS_FILE, 'wb') as f:
        pickle.dump(linear_runs, f)
    
    return linear_runs

def run_exponential_strategy_experiment():
    """Run experiments for different decay factors with exponential decay in DQN Mountain Car"""
    
    # Run experiments for each decay factor
    for seed in range(N_SEEDS):
        print(f"\n===== Seed {seed+1}/{N_SEEDS} =====")
        
        # Configuration 1 (epsilon_decay = 1.0)
        print(f"Running DQN with exponential decay, epsilon_decay=1.0")
        config1 = MOUNTAINCAR_CONFIG.copy()
        config1.update({
            "epsilon_decay_strategy": "exponential",
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "exploration_fraction": None,
            "epsilon_decay": 1.0
        })
        config1.update(EX3_DQN_MOUNTAINCAR_CONSTANTS)
        
        env = gym.make(config1["env"])
        eval_returns, eval_timesteps, times, run_data = train(env, config1, output=False)
        exp_run_10.update(eval_returns, eval_timesteps, times, run_data)
        env.close()
        
        # Configuration 2 (epsilon_decay = 0.5)
        print(f"Running DQN with exponential decay, epsilon_decay=0.5")
        config2 = MOUNTAINCAR_CONFIG.copy()
        config2.update({
            "epsilon_decay_strategy": "exponential",
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "exploration_fraction": None,
            "epsilon_decay": 0.5
        })
        config2.update(EX3_DQN_MOUNTAINCAR_CONSTANTS)
        
        env = gym.make(config2["env"])
        eval_returns, eval_timesteps, times, run_data = train(env, config2, output=False)
        exp_run_05.update(eval_returns, eval_timesteps, times, run_data)
        env.close()
        
        # Configuration 3 (epsilon_decay = 0.00001)
        print(f"Running DQN with exponential decay, epsilon_decay=0.00001")
        config3 = MOUNTAINCAR_CONFIG.copy()
        config3.update({
            "epsilon_decay_strategy": "exponential",
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "exploration_fraction": None,
            "epsilon_decay": 0.00001
        })
        config3.update(EX3_DQN_MOUNTAINCAR_CONSTANTS)
        
        env = gym.make(config3["env"])
        eval_returns, eval_timesteps, times, run_data = train(env, config3, output=False)
        exp_run_000001.update(eval_returns, eval_timesteps, times, run_data)
        env.close()

    # Collect all exponential decay runs
    exp_runs = [exp_run_10, exp_run_05, exp_run_000001]
    
    # Save results
    print(f"\nSaving exponential decay results to {EXP_SWEEP_RESULTS_FILE}")
    with open(EXP_SWEEP_RESULTS_FILE, 'wb') as f:
        pickle.dump(exp_runs, f)
    
    return exp_runs

def analyze_results(linear_runs, exp_runs):
    """Analyze and visualize the results for both decay strategies"""
    all_runs = linear_runs + exp_runs
    
    # Print performance statistics
    print("\n=== Performance Statistics ===")
    for run in all_runs:
        print(f"{run.run_name}:")
        print(f"  Mean final return: {run.final_return_mean:.4f} ± {run.final_return_ste:.4f}")

    # Rank all configurations
    ranked_runs = rank_runs(all_runs)
    print("\n=== All Configurations Ranked by Final Return ===")
    for i, run in enumerate(ranked_runs):
        print(f"{i+1}. {run.run_name}: {run.final_return_mean:.4f}")
    
    # Find best in each category
    ranked_linear = rank_runs(linear_runs)
    ranked_exp = rank_runs(exp_runs)
    
    best_linear = ranked_linear[0]
    best_exp = ranked_exp[0]
    
    print(f"\nBest linear decay strategy: {best_linear.run_name} with mean return {best_linear.final_return_mean:.4f}")
    print(f"Best exponential decay strategy: {best_exp.run_name} with mean return {best_exp.final_return_mean:.4f}")
    
    # Determine overall best strategy
    overall_best = ranked_runs[0]
    print(f"\nOverall best strategy: {overall_best.run_name} with mean return {overall_best.final_return_mean:.4f}")
    
    # Plot learning curves - Linear decay
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red']
    
    plt.subplot(1, 2, 1)
    for i, run in enumerate(linear_runs):
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
    plt.title('Linear Decay Strategies')
    plt.legend()
    plt.grid(True)

    # Plot learning curves - Exponential decay
    plt.subplot(1, 2, 2)
    for i, run in enumerate(exp_runs):
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
    plt.title('Exponential Decay Strategies')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_epsilon_decay_comparison.png')
    plt.show()
    
    # Calculate final epsilon values for questions 3.4 and 3.5
    # These assume running the full training process with max_timesteps from the config
    
    max_timesteps = EX3_DQN_MOUNTAINCAR_CONSTANTS["max_timesteps"]
    
    # For question 3.4: Exponential decay with decay=1.0
    # εₜ₊₁ ← r^(t/tₘₐₓ) * εₜ  (where r is decay factor)
    # Direct formula: ε = ε_start * (decay_factor ^ ((t*(t-1))/(2*t_max)))
    final_epsilon_q3_4 = 1.0 * (1.0 ** ((max_timesteps*(max_timesteps-1))/(2*max_timesteps)))
    final_epsilon_q3_4 = max(final_epsilon_q3_4, 0.05)  # Apply epsilon_min bound
    
    # For question 3.5: Exponential decay with decay=0.5
    final_epsilon_q3_5 = 1.0 * (0.5 ** ((max_timesteps*(max_timesteps-1))/(2*max_timesteps)))
    final_epsilon_q3_5 = max(final_epsilon_q3_5, 0.05)  # Apply epsilon_min bound
    
    print("\n=== Answers for Questions 3.4 and 3.5 ===")
    print(f"Question 3.4: Final epsilon with exponential decay=1.0: {final_epsilon_q3_4:.6f}")
    print(f"Question 3.5: Final epsilon with exponential decay=0.5: {final_epsilon_q3_5:.6f}")
    
    return overall_best.run_name, best_linear.run_name, best_exp.run_name, final_epsilon_q3_4, final_epsilon_q3_5

if __name__ == "__main__":
    import os
    
    # Check for existing linear decay results
    if os.path.exists(LINEAR_SWEEP_RESULTS_FILE):
        print(f"Loading existing linear decay results from {LINEAR_SWEEP_RESULTS_FILE}")
        with open(LINEAR_SWEEP_RESULTS_FILE, 'rb') as f:
            linear_runs = pickle.load(f)
    else:
        linear_runs = run_linear_strategy_experiment()
    
    # Check for existing exponential decay results
    if os.path.exists(EXP_SWEEP_RESULTS_FILE):
        print(f"Loading existing exponential decay results from {EXP_SWEEP_RESULTS_FILE}")
        with open(EXP_SWEEP_RESULTS_FILE, 'rb') as f:
            exp_runs = pickle.load(f)
    else:
        exp_runs = run_exponential_strategy_experiment()
    
    # Analyze all results
    best_overall, best_linear, best_exp, final_epsilon_q3_4, final_epsilon_q3_5 = analyze_results(linear_runs, exp_runs)
    
    # Print the answers for questions 3_2 and 3_3
    print("\nAnswer for question3_2:")
    print(f"The linear decay strategy with the highest mean return is: {best_linear}")
    
    print("\nAnswer for question3_3:")
    print(f"The exponential decay strategy with the highest mean return is: {best_exp}")
    
    print("\nOverall best strategy:")
    print(f"{best_overall}")
    
    # For question 3.6, the explanation would be written in the answer sheet:
    """
    Answer for question3_6:
    A strategy based on an exploration fraction parameter (like linear decay) is more generally 
    applicable across different environments than a decay strategy based on an epsilon decay parameter
    because it directly ties the exploration schedule to the expected training duration. This means:
    
    1) The exploration schedule automatically adapts to different training lengths without needing 
       parameter tuning
    2) It ensures that no matter how long or short the training process is, the agent will explore 
       the appropriate amount at the beginning and exploit more as training progresses
    3) It provides a more intuitive control over the exploration-exploitation trade-off, making it
       easier to reason about and adjust for different environments
    
    In contrast, a decay parameter must be carefully tuned based on the specific environment and
    training duration, requiring more experimentation to find the right value.
    """