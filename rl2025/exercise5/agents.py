import os
import gymnasium as gym
import numpy as np
from torch.optim import Adam
from typing import Dict, Iterable, List
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

from rl2025.exercise3.agents import Agent
from rl2025.exercise3.networks import FCNetwork
from rl2025.exercise3.replay import Transition


class DiagGaussian:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        eps = torch.randn(*self.mean.size())
        return self.mean + self.std * eps


class TD3Ensemble(Agent):
    """ TD3 with Ensemble of Critics
    
    An extension of TD3 that uses an ensemble of critics (more than 2) to further
    reduce overestimation bias and improve stability.
    
    :attr critics (List[FCNetwork]): ensemble of critic networks
    :attr critics_target (List[FCNetwork]): target networks for critics
    :attr critic_optim (torch.optim): PyTorch optimiser for all critics
    :attr actor (FCNetwork): fully connected actor network for policy
    :attr actor_target (FCNetwork): target network for actor
    :attr actor_optim (torch.optim): PyTorch optimiser for actor network
    :attr gamma (float): discount rate gamma
    :attr tau (float): soft update parameter
    :attr policy_delay (int): frequency of policy updates
    :attr policy_counter (int): counter for policy updates
    :attr target_noise (float): noise std for target policy smoothing
    :attr noise_clip (float): clipping value for target policy noise
    :attr num_critics (int): number of critics in the ensemble
    """

    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float,
            critic_learning_rate: float,
            policy_learning_rate: float,
            critic_hidden_size: Iterable[int],
            policy_hidden_size: Iterable[int],
            tau: float,
            policy_delay: int = 2,
            target_noise: float = 0.2,
            noise_clip: float = 0.5,
            num_critics: int = 4,  # ensemble size
            **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount rate gamma
        :param critic_learning_rate (float): learning rate for critic optimisation
        :param policy_learning_rate (float): learning rate for policy optimisation
        :param critic_hidden_size (Iterable[int]): list of hidden dimensionalities for critics
        :param policy_hidden_size (Iterable[int]): list of hidden dimensionalities for policy
        :param tau (float): step size for the update of the target networks
        :param policy_delay (int): number of critic updates per policy update
        :param target_noise (float): std of noise added to target policy
        :param noise_clip (float): limit for absolute value of target policy noise
        :param num_critics (int): number of critics in the ensemble (must be >= 2)
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        self.upper_action_bound = action_space.high[0]
        self.lower_action_bound = action_space.low[0]
        
        # Ensure we have at least 2 critics (TD3 minimum)
        self.num_critics = max(2, num_critics)

        # ######################################### #
        #  BUILD NETWORKS AND OPTIMIZERS           #
        # ######################################### #
        
        # Actor network and target
        self.actor = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )
        self.actor_target = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )
        self.actor_target.hard_update(self.actor)
        
        # Create ensemble of critics and their targets
        self.critics = []
        self.critics_target = []
        
        for i in range(self.num_critics):
            critic = FCNetwork(
                (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
            )
            critic_target = FCNetwork(
                (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
            )
            critic_target.hard_update(critic)
            
            self.critics.append(critic)
            self.critics_target.append(critic_target)

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        
        # Create a list of all critic parameters for the optimizer
        critic_parameters = []
        for critic in self.critics:
            critic_parameters.extend(list(critic.parameters()))
            
        self.critic_optim = Adam(critic_parameters, lr=critic_learning_rate, eps=1e-3)

        # ############################################# #
        # ALGORITHM HYPERPARAMETERS                    #
        # ############################################# #
        self.gamma = gamma
        self.critic_learning_rate = critic_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.tau = tau
        
        # TD3 specific parameters
        self.policy_delay = policy_delay
        self.policy_counter = 0  # Counter for policy delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        
        # Dynamic policy delay adjustment parameters
        self.base_policy_delay = policy_delay  # Store the base delay as reference
        self.min_policy_delay = 1              # Minimum policy delay
        self.max_policy_delay = 4              # Maximum policy delay
        self.critic_loss_history = []          # Store recent critic losses
        self.history_length = 20               # Number of recent losses to consider
        self.volatility_threshold = 0.2        # Threshold for determining volatility

        # ################################################### #
        # EXPLORATION NOISE                                   #
        # ################################################### #
        mean = torch.zeros(ACTION_SIZE)
        std = 0.1 * torch.ones(ACTION_SIZE)
        self.noise = DiagGaussian(mean, std)

        # Save network parameters for later restoration
        self.saveables = {
            "actor": self.actor,
            "actor_target": self.actor_target,
            "actor_optim": self.actor_optim,
            "critic_optim": self.critic_optim,
        }
        
        # Add all critics and their targets to saveables
        for i, (critic, critic_target) in enumerate(zip(self.critics, self.critics_target)):
            self.saveables[f"critic_{i}"] = critic
            self.saveables[f"critic_target_{i}"] = critic_target

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, filename: str, dir_path: str = None):
        """Restores PyTorch models from models file given by path
        
        :param filename (str): filename containing saved models
        :param dir_path (str, optional): path to directory where models file is located
        """
        if dir_path is None:
            dir_path = os.getcwd()
        save_path = os.path.join(dir_path, filename)
        
        checkpoint = torch.load(save_path)
        
        # Simply restore without any warnings about critic counts
        # Restore actor and actor target
        if "actor" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor"].state_dict())
        if "actor_target" in checkpoint:
            self.actor_target.load_state_dict(checkpoint["actor_target"].state_dict())
        
        # Restore optimizers if present
        if "actor_optim" in checkpoint:
            self.actor_optim.load_state_dict(checkpoint["actor_optim"].state_dict())
            
        if "critic_optim" in checkpoint:
            self.critic_optim.load_state_dict(checkpoint["critic_optim"].state_dict())
        
        # Restore critics and targets - only load the ones we have in our model
        for i in range(self.num_critics):
            critic_key = f"critic_{i}"
            target_key = f"critic_target_{i}"
            
            if critic_key in checkpoint:
                self.critics[i].load_state_dict(checkpoint[critic_key].state_dict())
            if target_key in checkpoint:
                self.critics_target[i].load_state_dict(checkpoint[target_key].state_dict())
        
        print(f"Successfully loaded model from {filename}")

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        # Convert observation to PyTorch tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        # Get deterministic action from actor network
        with torch.no_grad():
            action = self.actor(obs_tensor)
        
        # Convert action tensor to numpy array
        action = action.cpu().numpy().flatten()
        
        # Add noise for exploration if explore flag is True
        if explore:
            # Sample noise from the noise distribution
            noise = self.noise.sample().numpy()
            action = action + noise
        
        # Clip the action values to be within action space bounds
        action = np.clip(action, self.lower_action_bound, self.upper_action_bound)
        
        return action

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for TD3 with Ensemble Critics and Dynamic Policy Delay

        This function updates the ensemble of critics and the actor network with
        dynamically adjusted policy update frequency based on critic stability.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        states, actions, next_states, rewards, dones = batch
        
        # ---------------------- #
        # UPDATE CRITICS         #
        # ---------------------- #
        
        # Zero gradients
        self.critic_optim.zero_grad()
        
        # Concatenate states and actions for critic input
        state_action = torch.cat([states, actions], dim=1)
        
        # Compute target Q-values using target policy smoothing
        with torch.no_grad():
            # Get next actions from target actor
            next_actions = self.actor_target(next_states)
            
            # Add clipped noise to target actions for smoothing
            noise = torch.randn_like(next_actions) * self.target_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            smoothed_next_actions = torch.clamp(
                next_actions + noise, 
                self.lower_action_bound, 
                self.upper_action_bound
            )
            
            # Create input for target critics
            next_state_action = torch.cat([next_states, smoothed_next_actions], dim=1)
            
            # Get Q-values from all target critics
            next_q_values = []
            for critic_target in self.critics_target:
                next_q = critic_target(next_state_action)
                next_q_values.append(next_q)
            
            # Stack all Q-values and take the minimum across all critics
            next_q_values = torch.stack(next_q_values, dim=0)
            min_next_q = torch.min(next_q_values, dim=0)[0]
            
            # Calculate target Q-value using Bellman equation
            target_q = rewards + (1 - dones) * self.gamma * min_next_q
        
        # Compute critic losses for all critics
        critic_losses = []
        total_critic_loss = 0
        
        for i, critic in enumerate(self.critics):
            current_q = critic(state_action)
            critic_loss = F.mse_loss(current_q, target_q)
            critic_losses.append(critic_loss)
            total_critic_loss += critic_loss
        
        # Store the average critic loss for dynamic policy delay adjustment
        avg_critic_loss = total_critic_loss.item() / len(self.critics)
        self.critic_loss_history.append(avg_critic_loss)
        
        # Keep history at fixed length
        if len(self.critic_loss_history) > self.history_length:
            self.critic_loss_history.pop(0)
        
        # Dynamically adjust policy delay based on critic loss stability
        # Only do this when we have enough history
        if len(self.critic_loss_history) >= self.history_length:
            self._adjust_policy_delay()
            
        # Backpropagate the total loss through all critics
        total_critic_loss.backward()
        self.critic_optim.step()
        
        # ---------------------- #
        # DELAYED POLICY UPDATE  #
        # ---------------------- #
        
        # Only update policy and target networks at specified frequency
        policy_loss = 0.0
        self.policy_counter += 1
        
        if self.policy_counter >= self.policy_delay:
            self.policy_counter = 0
            
            # Zero policy gradient
            self.actor_optim.zero_grad()
            
            # Compute policy loss using all critics
            # The approach is to average the output of all critics for more stable policy updates
            policy_actions = self.actor(states)
            policy_state_action = torch.cat([states, policy_actions], dim=1)
            
            # Average Q-values across the first few critics (typically use critics 0 and 1)
            # This is to avoid overfitting to a single critic
            avg_q_value = self.critics[0](policy_state_action)
            for i in range(1, min(2, self.num_critics)):  # Use at most 2 critics for policy updates
                avg_q_value += self.critics[i](policy_state_action)
            avg_q_value /= min(2, self.num_critics)
            
            # Policy gradient is the negative of the Q-value
            policy_loss = -avg_q_value.mean()
            
            # Update actor
            policy_loss.backward()
            self.actor_optim.step()
            
            # ---------------------- #
            # UPDATE TARGET NETWORKS #
            # ---------------------- #
            
            # Soft update actor target
            self.actor_target.soft_update(self.actor, self.tau)
            
            # Soft update all critic targets
            for critic, critic_target in zip(self.critics, self.critics_target):
                critic_target.soft_update(critic, self.tau)
        
        # Prepare return dictionary with all losses
        return_dict = {
            "p_loss": policy_loss if isinstance(policy_loss, float) else policy_loss.item(),
            "total_critic_loss": total_critic_loss.item(),
            "policy_delay": self.policy_delay  # Include current policy delay
        }
        
        # Add individual critic losses
        for i, loss in enumerate(critic_losses):
            return_dict[f"critic_{i}_loss"] = loss.item()
        
        return return_dict
        
    def _adjust_policy_delay(self):
        """Dynamically adjusts policy delay based on critic loss stability.
        
        When critic losses are stable (low variance), we can update the policy more frequently.
        When critic losses are volatile (high variance), we should update the policy less frequently.
        """
        # Calculate volatility using coefficient of variation
        # (standard deviation / mean) of recent critic losses
        mean_loss = np.mean(self.critic_loss_history)
        if mean_loss > 0:  # Avoid division by zero
            std_loss = np.std(self.critic_loss_history)
            volatility = std_loss / mean_loss
            
            # Adjust policy delay based on volatility
            if volatility > self.volatility_threshold:
                # High volatility: Increase delay (slower policy updates)
                self.policy_delay = min(self.max_policy_delay, self.policy_delay + 1)
            else:
                # Low volatility: Decrease delay (faster policy updates)
                self.policy_delay = max(self.min_policy_delay, self.policy_delay - 1)

    
    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        # Adjust volatility threshold based on training progress
        # At the beginning, we expect more volatility, so we set a higher threshold
        # As training progresses, we become more sensitive to volatility
        progress = timestep / max_timesteps
        self.volatility_threshold = max(0.05, 0.2 - 0.15 * progress)
        
        # Optional: adjust noise parameters over time
        # self.target_noise = max(0.05, self.target_noise * (1 - 0.5 * progress))
        # self.noise_clip = max(0.1, self.noise_clip * (1 - 0.3 * progress))