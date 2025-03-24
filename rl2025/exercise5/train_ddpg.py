import copy
import pickle
from collections import defaultdict
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from rl2025.constants import EX4_RACETRACK_CONSTANTS as RACETRACK_CONSTANTS
from rl2025.exercise3.replay import ReplayBuffer
from rl2025.exercise3.networks import FCNetwork
from rl2025.exercise4.agents import DDPG, Agent

# Set up configuration
RENDER = False  # Set to True to visualize environment during evaluation
NUM_SEEDS = 5  # Number of seeds for statistical significance
RESULTS_DIR = "exercise5_results"
MODELS_DIR = "exercise5_models"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Environment
ENV = "racetrack-v0"

# Base configuration
CONFIG = {
    "critic_hidden_size": [32, 32, 32],
    "policy_hidden_size": [32, 32, 32],
    "vit_embed_dim": 128,
    "vit_depth": 4,
    "vit_heads": 4,
    "vit_output_dim": 64,
    "name": "ViT-DDPG",
}
CONFIG.update(RACETRACK_CONSTANTS)
CONFIG["save_filename"] = os.path.join(MODELS_DIR, "vit_ddpg.pt")

# Vision Transformer components
class SimpleSelfAttention(nn.Module):
    """Simple self-attention layer compatible with PyTorch 1.13.1"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projection to get Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        
        return out

class SimpleTransformerBlock(nn.Module):
    """Simplified transformer block compatible with PyTorch 1.13.1"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = SimpleSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class ViT_Base(nn.Module):
    """Simplified Vision Transformer base encoder for image processing"""
    def __init__(
        self, 
        image_size=96,
        patch_size=8, 
        in_channels=3, 
        embed_dim=128, 
        depth=4, 
        num_heads=4, 
        mlp_ratio=4.0, 
        dropout=0.1,
        output_dim=64
    ):
        super().__init__()
        
        # Handle patch size that doesn't divide image size
        while image_size % patch_size != 0 and patch_size > 1:
            patch_size -= 1
        
        if patch_size < 1:
            patch_size = 1
            
        print(f"Using patch size {patch_size} for image size {image_size}")
        
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding using convolution
        self.patch_embedding = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding and class token
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SimpleTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.head = nn.Linear(embed_dim, output_dim)
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embeddings
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Convert image to patches and embed
        x = self.patch_embedding(x)  # (batch_size, embed_dim, h', w')
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        
        # Add classification token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, 1 + n_patches, embed_dim)
        
        # Add position embedding (handle case where x has fewer patches than pos_embedding)
        pos_embed = self.pos_embedding[:, :x.size(1), :]
        x = x + pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Use the [CLS] token for classification
        x = x[:, 0]  # (batch_size, embed_dim)
        
        # Project to output dimension
        x = self.head(x)  # (batch_size, output_dim)
        
        return x

# Gaussian noise class for exploration
class DiagGaussian:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        eps = torch.randn(*self.mean.size())
        return self.mean + self.std * eps

# ViT-DDPG Agent class
class DDPG_ViT(DDPG):
    """DDPG with Vision Transformer for image observation processing"""
    
    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float,
            critic_learning_rate: float,
            policy_learning_rate: float,
            critic_hidden_size,
            policy_hidden_size,
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
        
        # Get observation dimensions
        if isinstance(observation_space, gym.spaces.Box):
            if len(observation_space.shape) == 3:  # Image: (H, W, C)
                IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = observation_space.shape
            elif len(observation_space.shape) == 1:  # Flattened
                # For the racetrack environment, let's use observation shape directly
                obs_dim = observation_space.shape[0]
                
                # Create a vector-based transformer instead
                IMG_HEIGHT = 1  # Treat as a single-pixel height
                IMG_WIDTH = obs_dim  # Width = observation dimension
                IMG_CHANNELS = 1  # Single channel
                
                print(f"Treating flat observation as 1x{obs_dim}x1 'image'")
            else:
                raise ValueError(f"Unsupported observation shape: {observation_space.shape}")
        else:
            raise ValueError(f"Unsupported observation space type: {type(observation_space)}")
        
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        
        # Create vision transformer for state encoding
        self.state_encoder = ViT_Base(
            image_size=max(IMG_HEIGHT, IMG_WIDTH),
            patch_size=2,  # Small patch size for vector observations
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
        
        # Add noise for exploration
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
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        # Reshape based on dimensionality
        if len(state.shape) == 1:  # Vector observation
            # Reshape to 1 x width x 1 "image"
            state = state.reshape(1, 1, self.IMG_WIDTH, 1).permute(0, 3, 1, 2)  # (B, C, H, W)
        elif len(state.shape) == 3:  # (H, W, C)
            state = state.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        elif len(state.shape) == 4:  # (B, H, W, C)
            state = state.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Normalize if needed
        if state.max() > 1.0:
            state = state / state.max()
        
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
    
    def update(self, batch):
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

# Helper functions for training and evaluation
def play_episode(
        env,
        agent,
        replay_buffer=None,
        train=False,
        explore=True,
        render=False,
        max_steps=200,
        batch_size=64,
):
    """Play one episode in the environment"""
    ep_data = defaultdict(list)
    obs, _ = env.reset()
    
    # Ensure observation format is consistent
    if isinstance(obs, np.ndarray):
        obs_formatted = obs.ravel()
    else:
        obs_formatted = obs
        
    done = False
    
    if render:
        env.render()

    episode_timesteps = 0
    episode_return = 0

    while not done and episode_timesteps < max_steps:
        # Get action from agent
        action = agent.act(obs_formatted, explore=explore)
        
        # Take action in environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Format next observation
        if isinstance(next_obs, np.ndarray):
            next_obs_formatted = next_obs.ravel()
        else:
            next_obs_formatted = next_obs
        
        # Store transition in replay buffer and update agent if training
        if train and replay_buffer is not None:
            replay_buffer.push(
                np.array(obs_formatted, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(next_obs_formatted, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                new_data = agent.update(batch)
                for k, v in new_data.items():
                    ep_data[k].append(v)

        # Update metrics
        episode_timesteps += 1
        episode_return += reward

        if render:
            env.render()

        obs = next_obs
        obs_formatted = next_obs_formatted

    return episode_timesteps, episode_return, ep_data

def train_and_evaluate(env, config, seed=None, output=True):
    """Train and evaluate the ViT-DDPG agent"""
    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
        if hasattr(env, 'seed'):
            env.seed(seed)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Initialize agent
    agent = DDPG_ViT(
        action_space=env.action_space,
        observation_space=env.observation_space,
        **config
    )
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(config["buffer_capacity"])
    
    # Initialize tracking variables
    timesteps_elapsed = 0
    eval_returns_all = []
    eval_timesteps_all = []
    eval_times_all = []
    training_losses = {"q_loss": [], "p_loss": []}
    run_data = defaultdict(list)
    
    # Start timer
    start_time = time.time()
    
    # Training loop
    with tqdm(total=config["max_timesteps"], desc=f"Training {config['name']} (seed {seed})") as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            # Check if we've exceeded maximum training time
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                if output:
                    pbar.write(f"Training ended after {elapsed_seconds:.2f}s")
                break
            
            # Play an episode for training
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
            
            # Update tracking variables
            timesteps_elapsed += episode_timesteps
            pbar.update(episode_timesteps)
            
            # Track training metrics
            if 'q_loss' in ep_data and ep_data['q_loss']:
                training_losses['q_loss'].extend(ep_data['q_loss'])
            if 'p_loss' in ep_data and ep_data['p_loss']:
                training_losses['p_loss'].extend(ep_data['p_loss'])
                
            for k, v in ep_data.items():
                run_data[k].extend(v)
            run_data["train_ep_returns"].append(ep_return)
            
            # Periodic evaluation
            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                # Create evaluation environment
                eval_env = gym.make(config["env"])
                if seed is not None and hasattr(eval_env, 'seed'):
                    eval_env.seed(seed + 1000)  # Different seed for evaluation
                
                eval_returns = 0
                episodic_returns = []
                
                # Run several evaluation episodes
                for eval_ep in range(config["eval_episodes"]):
                    _, episode_return, _ = play_episode(
                        eval_env,
                        agent,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=config["episode_length"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]
                    episodic_returns.append(episode_return)
                    
                    # More verbose output per evaluation episode
                    if output:
                        print(f"  Eval episode {eval_ep+1}/{config['eval_episodes']}: Return = {episode_return:.2f}")
                
                # Log evaluation results
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed}: "
                        f"mean return={eval_returns:.2f}, "
                        f"min={min(episodic_returns):.2f}, "
                        f"max={max(episodic_returns):.2f}"
                    )
                
                # Update metrics
                eval_returns_all.append(eval_returns)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(elapsed_seconds)
                
                # Early stopping if target return reached
                if eval_returns >= config["target_return"]:
                    if output:
                        pbar.write(
                            f"Reached return {eval_returns:.2f} >= "
                            f"target return of {config['target_return']:.2f}"
                        )
                    break
                
                # Close evaluation environment
                eval_env.close()
    
    # Save model if requested
    if config["save_filename"]:
        save_path = agent.save(config["save_filename"])
        if output:
            print(f"Saved model to: {save_path}")
    
    # Return training metrics
    metrics = {
        "returns": np.array(eval_returns_all),
        "timesteps": np.array(eval_timesteps_all),
        "times": np.array(eval_times_all),
        "training_losses": training_losses,
        "training_time": elapsed_seconds,
        "final_return": eval_returns_all[-1] if eval_returns_all else 0.0,
        "peak_return": max(eval_returns_all) if eval_returns_all else 0.0,
        "run_data": run_data
    }
    
    return metrics

# Main experiment function
def run_experiment():
    """Run the ViT-DDPG experiment with multiple seeds"""
    # Results dictionary
    all_metrics = []
    
    # Run experiment with multiple seeds
    for seed in range(NUM_SEEDS):
        print(f"\n===== Running seed {seed+1}/{NUM_SEEDS} =====")
        
        # Create environment
        env = gym.make(ENV)
        
        # Train and evaluate
        metrics = train_and_evaluate(env, CONFIG.copy(), seed=seed)
        all_metrics.append(metrics)
        
        # Close environment
        env.close()
    
    # Calculate aggregate statistics
    mean_returns = np.mean([metrics["returns"] for metrics in all_metrics], axis=0)
    std_returns = np.std([metrics["returns"] for metrics in all_metrics], axis=0)
    mean_timesteps = np.mean([metrics["timesteps"] for metrics in all_metrics], axis=0)
    final_returns = [metrics["final_return"] for metrics in all_metrics]
    training_times = [metrics["training_time"] for metrics in all_metrics]
    
    # Compile results
    results = {
        "mean_returns": mean_returns,
        "std_returns": std_returns,
        "mean_timesteps": mean_timesteps,
        "final_returns": final_returns,
        "mean_final_return": np.mean(final_returns),
        "std_final_return": np.std(final_returns),
        "mean_training_time": np.mean(training_times),
        "std_training_time": np.std(training_times),
        "all_metrics": all_metrics
    }
    
    # Save results
    with open(os.path.join(RESULTS_DIR, "vit_ddpg_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    return results

# Plotting function
def plot_results(results):
    """Plot the results of the ViT-DDPG experiment"""
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot learning curve
    ax = axes[0]
    ax.plot(results["mean_timesteps"], results["mean_returns"], color="blue", label="ViT-DDPG")
    ax.fill_between(
        results["mean_timesteps"],
        results["mean_returns"] - results["std_returns"],
        results["mean_returns"] + results["std_returns"],
        color="blue",
        alpha=0.2
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Evaluation Return")
    ax.set_title("ViT-DDPG Learning Curve")
    ax.legend()
    ax.grid(True)
    
    # Plot training losses if available
    ax = axes[1]
    
    # Average Q-losses and policy losses across seeds
    q_losses = []
    p_losses = []
    
    for metrics in results["all_metrics"]:
        if "training_losses" in metrics and "q_loss" in metrics["training_losses"]:
            q_losses.append(metrics["training_losses"]["q_loss"])
        if "training_losses" in metrics and "p_loss" in metrics["training_losses"]:
            p_losses.append(metrics["training_losses"]["p_loss"])
    
    if q_losses and p_losses:
        # Get minimum length for averaging
        min_q_len = min(len(losses) for losses in q_losses)
        min_p_len = min(len(losses) for losses in p_losses)
        
        # Truncate to minimum length
        q_losses_aligned = [losses[:min_q_len] for losses in q_losses]
        p_losses_aligned = [losses[:min_p_len] for losses in p_losses]
        
        # Compute mean and std
        mean_q_loss = np.mean(q_losses_aligned, axis=0)
        std_q_loss = np.std(q_losses_aligned, axis=0)
        mean_p_loss = np.mean(p_losses_aligned, axis=0)
        std_p_loss = np.std(p_losses_aligned, axis=0)
        
        # Plot moving averages for smoother curves
        window_size = min(100, min(min_q_len, min_p_len) // 10)
        
        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        if min_q_len > window_size:
            x_q = np.arange(mean_q_loss.size - window_size + 1)
            mean_q_loss_smooth = moving_average(mean_q_loss, window_size)
            ax.plot(x_q, mean_q_loss_smooth, color="red", label="Q-Loss")
        
        if min_p_len > window_size:
            x_p = np.arange(mean_p_loss.size - window_size + 1)
            mean_p_loss_smooth = moving_average(mean_p_loss, window_size)
            ax.plot(x_p, mean_p_loss_smooth, color="green", label="Policy Loss")
        
        ax.set_xlabel("Update Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Training Losses")
        ax.legend()
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "No loss data available", ha="center", va="center", transform=ax.transAxes)
    
    # Add a title for the entire figure
    fig.suptitle("Vision Transformer DDPG Performance", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    
    # Save the figure
    plt.savefig(os.path.join(RESULTS_DIR, "vit_ddpg_performance.png"), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, "vit_ddpg_performance.pdf"))
    
 # Print summary statistics
    print("\n===== ViT-DDPG Performance Summary =====")
    print(f"Mean Final Return: {results['mean_final_return']:.2f} ± {results['std_final_return']:.2f}")
    print(f"Mean Training Time: {results['mean_training_time']:.2f}s")
    print(f"Best Seed Final Return: {max(results['final_returns']):.2f}")
    
    plt.show()

# Entry point
if __name__ == "__main__":
    # Run experiment
    results = run_experiment()
    
    # Plot results
    plot_results(results)