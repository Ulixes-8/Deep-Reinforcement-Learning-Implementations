import os
import gymnasium as gym
import numpy as np
from torch.optim import Adam
from typing import Dict, Iterable
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


class TD3(Agent):
    """ TD3 (Twin Delayed DDPG)
    
    An extension of DDPG that addresses function approximation errors by:
    1. Using twin critics to reduce overestimation bias
    2. Delayed policy updates for stability
    3. Target policy smoothing via noise addition
    
    :attr critic1 (FCNetwork): first fully connected critic network
    :attr critic2 (FCNetwork): second fully connected critic network
    :attr critic1_target (FCNetwork): target network for first critic
    :attr critic2_target (FCNetwork): target network for second critic
    :attr critic_optim (torch.optim): PyTorch optimiser for both critics
    :attr actor (FCNetwork): fully connected actor network for policy
    :attr actor_target (FCNetwork): target network for actor
    :attr actor_optim (torch.optim): PyTorch optimiser for actor network
    :attr gamma (float): discount rate gamma
    :attr tau (float): soft update parameter
    :attr policy_delay (int): frequency of policy updates
    :attr policy_counter (int): counter for policy updates
    :attr target_noise (float): noise std for target policy smoothing
    :attr noise_clip (float): clipping value for target policy noise
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
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        self.upper_action_bound = action_space.high[0]
        self.lower_action_bound = action_space.low[0]

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
        
        # Twin critics and targets
        self.critic1 = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic1_target = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic1_target.hard_update(self.critic1)
        
        self.critic2 = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic2_target = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic2_target.hard_update(self.critic2)

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        self.critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optim = Adam(self.critic_params, lr=critic_learning_rate, eps=1e-3)

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

        # ################################################### #
        # EXPLORATION NOISE                                   #
        # ################################################### #
        mean = torch.zeros(ACTION_SIZE)
        std = 0.1 * torch.ones(ACTION_SIZE)
        self.noise = DiagGaussian(mean, std)

        # Save network parameters for later restoration
        self.saveables.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "critic1": self.critic1,
                "critic1_target": self.critic1_target,
                "critic2": self.critic2,
                "critic2_target": self.critic2_target,
                "actor_optim": self.actor_optim,
                "critic_optim": self.critic_optim,
            }
        )

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
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

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
        """Update function for TD3

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your critics and delayed updates for the actor.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        states, actions, next_states, rewards, dones = batch
        
        # ---------------------- #
        # UPDATE CRITICS         #
        # ---------------------- #
        
        # Zero gradients
        self.critic_optim.zero_grad()
        
        # Compute current Q-values for both critics
        state_action = torch.cat([states, actions], dim=1)
        current_q1 = self.critic1(state_action)
        current_q2 = self.critic2(state_action)
        
        # Compute target Q-values using target policy smoothing
        with torch.no_grad():
            # Get next actions from target actor
            next_actions = self.actor_target(next_states)
            
            # Add clipped noise to target actions for smoothing (TD3 improvement 1)
            noise = torch.randn_like(next_actions) * self.target_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            smoothed_next_actions = torch.clamp(
                next_actions + noise, 
                self.lower_action_bound, 
                self.upper_action_bound
            )
            
            # Get Q-values from both target critics for the smoothed next actions
            next_state_action = torch.cat([next_states, smoothed_next_actions], dim=1)
            next_q1 = self.critic1_target(next_state_action)
            next_q2 = self.critic2_target(next_state_action)
            
            # Take the minimum of both Q-values (TD3 improvement 2)
            next_q = torch.min(next_q1, next_q2)
            
            # Calculate target Q-value using Bellman equation
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute critic losses (MSE)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        # Update critics
        critic_loss.backward()
        self.critic_optim.step()
        
        # ---------------------- #
        # DELAYED POLICY UPDATE  #
        # ---------------------- #
        
        # Only update policy and target networks at specified frequency (TD3 improvement 3)
        policy_loss = 0.0
        self.policy_counter += 1
        
        if self.policy_counter >= self.policy_delay:
            self.policy_counter = 0
            
            # Zero policy gradient
            self.actor_optim.zero_grad()
            
            # Compute policy loss using only the first critic for stable policy updates
            policy_actions = self.actor(states)
            policy_state_action = torch.cat([states, policy_actions], dim=1)
            policy_loss = -self.critic1(policy_state_action).mean()
            
            # Update policy
            policy_loss.backward()
            self.actor_optim.step()
            
            # ---------------------- #
            # UPDATE TARGET NETWORKS #
            # ---------------------- #
            
            # Soft update of all target networks
            self.critic1_target.soft_update(self.critic1, self.tau)
            self.critic2_target.soft_update(self.critic2, self.tau)
            self.actor_target.soft_update(self.actor, self.tau)
        
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "p_loss": policy_loss if isinstance(policy_loss, float) else policy_loss.item(),
            "critic_loss": critic_loss.item()
        }
    
    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        **YOU MAY IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        pass