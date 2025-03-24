# Reinforcement Learning 2025 Repository

This repository contains comprehensive implementations of various reinforcement learning algorithms developed as coursework for the Reinforcement Learning course at the University of Edinburgh. The codebase showcases a wide range of RL techniques, from foundational methods to advanced deep reinforcement learning approaches.

## Repository Structure

The code is organized into five main exercises, each exploring different aspects of reinforcement learning:

### Exercise 1: MDP Solvers
- **Value Iteration**: Implementation of the classic value iteration algorithm for solving Markov Decision Processes.
- **Policy Iteration**: Implementation of policy iteration with policy evaluation and policy improvement steps.
- **Run**: Execute `mdp_solver.py` to see these algorithms in action on a simple rock-jumping MDP.

### Exercise 2: Model-Free RL
- **Q-Learning**: Implementation of tabular Q-learning with epsilon-greedy action selection.
- **Monte Carlo Control**: Implementation of every-visit Monte Carlo control for episodic environments.
- **Environment**: Both algorithms are tested on the FrozenLake-8x8 environment (slippery and non-slippery variants).
- **Run**: Execute `train_q_learning.py` or `train_monte_carlo.py` to train and evaluate the agents.

### Exercise 3: Deep Q-Learning
- **DQN**: Implementation of Deep Q-Network with experience replay and target networks.
- **DiscreteRL**: A tabular Q-learning approach with state discretization for continuous state spaces.
- **Epsilon Decay Strategies**: Linear and exponential decay strategies for exploration.
- **Environments**: CartPole-v1 and MountainCar-v0 environments.
- **Run**: Execute `train_dqn.py` or `train_discreterl.py` with different hyperparameter configurations.

### Exercise 4: Deep Deterministic Policy Gradient (DDPG)
- **DDPG**: Implementation of DDPG for continuous action spaces.
- **Highway Environment**: Trained and evaluated on the racetrack-v0 environment.
- **Run**: Execute `train_ddpg.py` to train the agent and `evaluate_ddpg.py` to evaluate its performance.

### Exercise 5: TD3 with Ensemble Critics and Dynamic Policy Delay (Novel Approach)
- **TD3Ensemble**: A novel extension of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.
- **Key Innovations**:
  1. **Critic Ensemble**: Uses an ensemble of four critics instead of just twins for more robust value estimation.
  2. **Dynamic Policy Delay**: Automatically adjusts the frequency of policy updates based on critic learning stability.
  3. **Adaptive Volatility Threshold**: Becomes increasingly sensitive to critic volatility as training progresses.
  4. **Selective Critic Utilization**: Strategically selects which critics to use for policy updates.
- **Performance**: Achieved exceptional performance in the Highway-v0 environment with a mean max return of 996.41.
- **Run**: Execute `train_td3.py` to train the agent and `evaluate_td3.py` to evaluate its performance.

## Environment Setup

To get started with this codebase, follow these steps:

1. **Install Anaconda**
   - Navigate to the [Anaconda download page](https://www.anaconda.com/products/distribution) and follow the installation instructions for your operating system.
   - Anaconda (sometimes shortened to "conda") supports Linux, MacOS, and Windows.

2. **Create a conda environment**
   ```bash
   # Create a new conda environment with Python 3.7
   conda create -n rl_course python=3.7
   
   # Activate the conda environment
   conda activate rl_course
   ```

3. **Download and install the codebase**
   ```bash
   # After unzipping rl2025-coursework.zip, navigate to the coursework folder
   cd rl2025-coursework
   
   # Install the required packages using pip
   pip install -e .
   ```
   This will install all the required dependencies specified in the `setup.py` file.

## Running the Code

Navigate to the appropriate directories before running each file:

### Exercise 1: MDP Solvers
```bash
# Navigate to the exercise1 directory
cd rl2025/exercise1

# Run the MDP solver
python mdp_solver.py
```

### Exercise 2: Model-Free RL
```bash
# Navigate to the exercise2 directory
cd rl2025/exercise2

# Run Q-Learning
python train_q_learning.py

# Run Monte Carlo Control
python train_monte_carlo.py
```

### Exercise 3: Deep Q-Learning
```bash
# Navigate to the exercise3 directory
cd rl2025/exercise3

# Run DQN
python train_dqn.py

# Run DiscreteRL
python train_discreterl.py

# Evaluate trained agents
python evaluate_dqn.py
python evaluate_discreterl.py
```

### Exercise 4: DDPG
```bash
# Navigate to the exercise4 directory
cd rl2025/exercise4

# Train DDPG
python train_ddpg.py

# Evaluate DDPG
python evaluate_ddpg.py
```

### Exercise 5: TD3Ensemble (Novel Approach)
```bash
# Navigate to the exercise5 directory
cd rl2025/exercise5

# Train TD3Ensemble
python train_td3.py

# Evaluate TD3Ensemble
python evaluate_td3.py
```

## TD3Ensemble with Dynamic Policy Delay: Novel Contribution

The innovative TD3Ensemble implementation (Exercise 5) represents the most significant contribution of this repository. This approach extends the standard TD3 algorithm with:

1. **Larger Critic Ensemble**: Employs 4 critics rather than the standard 2, providing more robust value estimation and reducing overestimation bias.

2. **Dynamic Policy Delay Adjustment**: Unlike standard TD3 which uses a fixed policy delay, TD3Ensemble dynamically adjusts the frequency of policy updates based on critic learning stability. When critic loss volatility is high, policy updates become less frequent; when stability improves, updates occur more frequently.

3. **Adaptive Volatility Detection**: Uses a training-progress-aware volatility threshold that becomes increasingly strict as training progresses, allowing for more exploration early on and more exploitation later.

4. **Selective Critic Utilization**: Uses a subset of critics for policy updates to prevent overfitting to any single critic's estimates.

This novel approach significantly outperforms baseline TD3 in the Highway-v0 environment, achieving an average max return of approximately 996 compared to the baseline's 552.

## Evaluation Methodology

For Exercise 5, the evaluation methodology involves 10 independent runs of 3 episodes each, reporting the maximum return per run and calculating the mean across all runs. This process is repeated four times to ensure reproducibility, with consistent results showing returns between 800-1300 in optimal episodes.

## Acknowledgments

This repository contains coursework completed for the Reinforcement Learning course at the University of Edinburgh. Implementation details and experimental designs follow the course guidelines while incorporating novel elements, particularly in Exercise 5.
