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
from rl2025.exercise5.vision_transformer import ViT_Base
from rl2025.exercise4.agents import DDPG

class DiagGaussian(torch.nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        eps = Variable(torch.randn(*self.mean.size()))
        return self.mean + self.std * eps

class DDPG_ViT(DDPG):
    """DDPG with Vision Transformer for image observation processing"""
    
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
            vit_embed_dim=128,
            vit_depth=4,
            vit_heads=4,
            vit_output_dim=64,
            **kwargs,
    ):
        # Initialize parent class but we'll override networks
        super().__init__(
            action_space, observation_space, gamma, 
            critic_learning_rate, policy_learning_rate,
            critic_hidden_size, policy_hidden_size, tau, **kwargs
        )
        
        # Get dimensions
        ACTION_SIZE = action_space.shape[0]
        
        # Determine image dimensions from observation space
        if isinstance(observation_space, gym.spaces.Box):
            if len(observation_space.shape) == 3:  # (H, W, C)
                IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = observation_space.shape
            elif len(observation_space.shape) == 1:  # Flattened image
                # Try to infer the original image shape
                # This depends on knowing the expected image dimensions
                # For highway-env/racetrack, let's use a common resolution
                flat_dim = observation_space.shape[0]
                # Try to determine if it's a square image
                for i in range(1, 101):  # Try square sizes up to 100x100
                    if flat_dim % (i*i) == 0:
                        IMG_HEIGHT = IMG_WIDTH = i
                        IMG_CHANNELS = flat_dim // (i*i)
                        if IMG_CHANNELS in [1, 3, 4]:  # Common channel counts
                            break
                else:
                    # If we couldn't determine a square image, use a default
                    print("Warning: Could not determine original image dimensions. Using defaults.")
                    IMG_HEIGHT = IMG_WIDTH = 96  # Default
                    IMG_CHANNELS = 3  # Default
            else:
                raise ValueError(f"Unexpected observation shape: {observation_space.shape}")
        else:
            raise ValueError(f"Unexpected observation space type: {type(observation_space)}")
        
        print(f"Using image dimensions: {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}")
        
        # Create vision transformer for state encoding
        self.state_encoder = ViT_Base(
            image_size=IMG_HEIGHT,  # assuming square image
            patch_size=8,  # will be adjusted if needed
            in_channels=IMG_CHANNELS,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            output_dim=vit_output_dim
        )
        
        # Create actor network (policy)
        self.actor = FCNetwork(
            (vit_output_dim, *policy_hidden_size, ACTION_SIZE), 
            output_activation=torch.nn.Tanh
        )
        
        # Create actor target network
        self.actor_target = FCNetwork(
            (vit_output_dim, *policy_hidden_size, ACTION_SIZE), 
            output_activation=torch.nn.Tanh
        )
        self.actor_target.hard_update(self.actor)
        
        # Create critic network (Q-function)
        self.critic = FCNetwork(
            (vit_output_dim + ACTION_SIZE, *critic_hidden_size, 1), 
            output_activation=None
        )
        
        # Create critic target network
        self.critic_target = FCNetwork(
            (vit_output_dim + ACTION_SIZE, *critic_hidden_size, 1), 
            output_activation=None
        )
        self.critic_target.hard_update(self.critic)
        
        # Create optimizers
        self.state_encoder_optim = torch.optim.Adam(
            self.state_encoder.parameters(), 
            lr=critic_learning_rate, 
            eps=1e-3
        )
        self.policy_optim = torch.optim.Adam(
            self.actor.parameters(), 
            lr=policy_learning_rate, 
            eps=1e-3
        )
        self.critic_optim = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.state_encoder.parameters()),
            lr=critic_learning_rate, 
            eps=1e-3
        )
        
        # Add noise for exploration (as defined in the DDPG paper)
        self.noise = DiagGaussian(torch.zeros(ACTION_SIZE), 0.1 * torch.ones(ACTION_SIZE))
        
        # Update saveables
        self.saveables.update({
            "state_encoder": self.state_encoder,
            "actor": self.actor,
            "actor_target": self.actor_target,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "state_encoder_optim": self.state_encoder_optim,
            "policy_optim": self.policy_optim,
            "critic_optim": self.critic_optim,
        })
    
    def _encode_state(self, state):
        """Encode state using vision transformer"""
        # Handle different input formats
        if isinstance(state, np.ndarray):
            if len(state.shape) == 1:
                # If we get a flat array, reshape it to an image
                # We need to know the expected dimensions
                # For example, if we know it's a 96x96x3 image flattened:
                state = state.reshape(96, 96, 3)
            
            # Convert numpy array to tensor
            state = torch.FloatTensor(state)
        
        # Reshape to match expected input: (batch_size, channels, height, width)
        if len(state.shape) == 3:  # (height, width, channels)
            state = state.permute(2, 0, 1).unsqueeze(0)  # (1, channels, height, width)
        elif len(state.shape) == 4:  # (batch_size, height, width, channels)
            state = state.permute(0, 3, 1, 2)  # (batch_size, channels, height, width)
        
        # Normalize pixel values to [0, 1]
        if state.max() > 1.0:
            state = state / 255.0
        
        # Encode state
        encoded_state = self.state_encoder(state)
        return encoded_state
    
    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)"""
        # Convert observation to tensor and encode it
        with torch.no_grad():
            encoded_state = self._encode_state(obs)
            action = self.actor(encoded_state)
        
        # Convert to numpy
        action = action.cpu().numpy().flatten()
        
        # Add noise for exploration
        if explore:
            noise = self.noise.sample().numpy()
            action = action + noise
        
        # Clip action to be within action space bounds
        action = np.clip(action, self.lower_action_bound, self.upper_action_bound)
        
        return action
    
    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DDPG with vision transformer"""
        states, actions, next_states, rewards, dones = batch
        
        # Encode states and next states
        encoded_states = self._encode_state(states)
        encoded_next_states = self._encode_state(next_states)
        
        # Update critic
        self.critic_optim.zero_grad()
        
        # Current Q-values
        state_action = torch.cat([encoded_states, actions], dim=1)
        current_q = self.critic(state_action)
        
        # Target Q-values
        with torch.no_grad():
            next_actions = self.actor_target(encoded_next_states)
            next_state_action = torch.cat([encoded_next_states, next_actions], dim=1)
            next_q = self.critic_target(next_state_action)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute critic loss
        q_loss = F.mse_loss(current_q, target_q)
        q_loss.backward()
        self.critic_optim.step()
        
        # Update actor
        self.policy_optim.zero_grad()
        
        # Get current actions according to policy
        current_actions = self.actor(encoded_states.detach())
        
        # Compute policy loss
        state_action = torch.cat([encoded_states.detach(), current_actions], dim=1)
        p_loss = -self.critic(state_action).mean()
        
        p_loss.backward()
        self.policy_optim.step()
        
        # Update target networks
        self.critic_target.soft_update(self.critic, self.tau)
        self.actor_target.soft_update(self.actor, self.tau)
        
        return {
            "q_loss": q_loss.item(),
            "p_loss": p_loss.item(),
        }
    

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