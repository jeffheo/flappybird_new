import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
import time
from datetime import datetime

class MetricsTracker:
    """
    Tracks reward, loss values, exploration vs. exploitation, and model parameter statistics.
    Separates metrics by their time scales (episode-based vs update-based).
    """
    def __init__(self, agent_name, window_size=100, save_dir="metrics"):
        self.agent_name = agent_name
        self.window_size = window_size
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.rewards_window = deque(maxlen=window_size)
        
        self.losses = {
            'total': [],
            'policy': [],
            'value': [],
            'entropy': [],
            'td_error': []
        }
        
        self.exploration_values = []  # epsilon for DQN, entropy for PPO
        
        self.param_stats = {
            'grad_norm': [],
            'weight_norm': [],
            'update_ratio': [],  
            'clipped_updates': []  
        }
        
        self.update_steps = []
        self.current_update_step = 0
        
        self.start_time = time.time()
        self.episode_start_time = None
        
        self.episode_metrics_df = None
        self.training_metrics_df = None
    
    def start_episode(self):
        self.episode_start_time = time.time()
    
    def end_episode(self, reward, length):
        
        if self.episode_start_time is None:
            episode_time = 0
        else:
            episode_time = time.time() - self.episode_start_time
        
        self.episode_rewards.append(reward)
        self.rewards_window.append(reward)
        self.episode_lengths.append(length)
        self.episode_times.append(episode_time)
    
    def log_loss(self, total=None, policy=None, value=None, entropy=None, td_error=None):
        
        self.current_update_step += 1
        self.update_steps.append(self.current_update_step)
        
        if total is not None:
            self.losses['total'].append(total)
        if policy is not None:
            self.losses['policy'].append(policy)
        if value is not None:
            self.losses['value'].append(value)
        if entropy is not None:
            self.losses['entropy'].append(entropy)
        if td_error is not None:
            self.losses['td_error'].append(td_error)
    
    def log_exploration(self, value):
        
        self.exploration_values.append(value)
    
    def log_param_stats(self, model, grad_norm=None, update_ratio=None, clipped_updates=None):
        
        if grad_norm is None and model is not None:
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
        
        if grad_norm is not None:
            self.param_stats['grad_norm'].append(grad_norm)
        
        if model is not None:
            weight_norm = 0
            for param in model.parameters():
                weight_norm += param.data.norm(2).item() ** 2
            weight_norm = weight_norm ** 0.5
            self.param_stats['weight_norm'].append(weight_norm)
        
        if update_ratio is not None:
            self.param_stats['update_ratio'].append(update_ratio)
        if clipped_updates is not None:
            self.param_stats['clipped_updates'].append(clipped_updates)
    
    def get_current_stats(self):
        
        stats = {
            'episode': len(self.episode_rewards),
            'reward_mean': np.mean(self.rewards_window) if self.rewards_window else 0,
            'reward_last': self.episode_rewards[-1] if self.episode_rewards else 0,
            'reward_max': max(self.episode_rewards) if self.episode_rewards else 0,
            'episode_length_mean': np.mean(self.episode_lengths[-self.window_size:]) if self.episode_lengths else 0,
            'time_elapsed': time.time() - self.start_time
        }
        
        for loss_type, values in self.losses.items():
            if values:
                stats[f'{loss_type}_loss_mean'] = np.mean(values[-self.window_size:])
        
        if self.exploration_values:
            stats['exploration'] = self.exploration_values[-1]
        
        for stat_type, values in self.param_stats.items():
            if values:
                stats[f'{stat_type}_mean'] = np.mean(values[-self.window_size:])
        
        return stats
    
    def create_episode_dataframe(self):
        data = {
            'episode': list(range(1, len(self.episode_rewards) + 1)),
            'reward': self.episode_rewards,
            'episode_length': self.episode_lengths,
            'episode_time': self.episode_times
        }
        
        window = min(self.window_size, len(self.episode_rewards))
        if window > 0:
            data['reward_moving_avg'] = pd.Series(self.episode_rewards).rolling(window=window).mean().values
            data['length_moving_avg'] = pd.Series(self.episode_lengths).rolling(window=window).mean().values
        
        self.episode_metrics_df = pd.DataFrame(data)
        return self.episode_metrics_df
    
    def create_training_dataframe(self):
        
        has_training_metrics = (
            any(len(values) > 0 for values in self.losses.values()) or 
            len(self.exploration_values) > 0 or
            any(len(values) > 0 for values in self.param_stats.values())
        )
        
        if not has_training_metrics:
            return None
        
        max_updates = max(
            max([len(values) for values in self.losses.values()] or [0]),
            len(self.exploration_values),
            max([len(values) for values in self.param_stats.values()] or [0])
        )
        
        if max_updates == 0:
            return None
            
        data = {
            'update_step': list(range(1, max_updates + 1))
        }
        
        for loss_type, values in self.losses.items():
            if values:
                padded_values = values + [np.nan] * (max_updates - len(values))
                data[f'{loss_type}_loss'] = padded_values
        
        if self.exploration_values:
            padded_values = self.exploration_values + [np.nan] * (max_updates - len(self.exploration_values))
            data['exploration'] = padded_values
        
        for stat_type, values in self.param_stats.items():
            if values:
                padded_values = values + [np.nan] * (max_updates - len(values))
                data[stat_type] = padded_values
        
        self.training_metrics_df = pd.DataFrame(data)
        return self.training_metrics_df
    
    def save_metrics(self, additional_info=None, generate_plots=False):
        
        filename_base = f"{self.agent_name}_{self.timestamp}"
        if additional_info:
            for key, value in additional_info.items():
                filename_base += f"_{key}_{value}"
        
        saved_paths = {}
        
        episode_df = self.create_episode_dataframe()
        if episode_df is not None and not episode_df.empty:
            episode_csv_path = os.path.join(self.save_dir, f"{filename_base}_episodes.csv")
            episode_df.to_csv(episode_csv_path, index=False)
            saved_paths['episodes'] = episode_csv_path
        
        training_df = self.create_training_dataframe()
        if training_df is not None and not training_df.empty:
            training_csv_path = os.path.join(self.save_dir, f"{filename_base}_training.csv")
            training_df.to_csv(training_csv_path, index=False)
            saved_paths['training'] = training_csv_path
        
        if generate_plots:
            self.plot_metrics(filename_base)
        
        return saved_paths
    
    def plot_metrics(self, filename_base=None):
       
        if filename_base is None:
            filename_base = f"{self.agent_name}_{self.timestamp}"
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.episode_rewards, alpha=0.3, color='blue', label='Rewards')
        
        if len(self.episode_rewards) >= self.window_size:
            moving_avg = np.convolve(self.episode_rewards, 
                                     np.ones(self.window_size)/self.window_size, 
                                     mode='valid')
            plt.plot(range(self.window_size-1, len(self.episode_rewards)), 
                     moving_avg, color='red', 
                     label=f'{self.window_size}-episode Moving Average')
        
        plt.title(f'{self.agent_name} Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(self.episode_lengths, alpha=0.3, color='green', label='Episode Length')
        
        if len(self.episode_lengths) >= self.window_size:
            moving_avg = np.convolve(self.episode_lengths, 
                                     np.ones(self.window_size)/self.window_size, 
                                     mode='valid')
            plt.plot(range(self.window_size-1, len(self.episode_lengths)), 
                     moving_avg, color='orange', 
                     label=f'{self.window_size}-episode Moving Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{filename_base}_rewards.png"))
        plt.close()
        
        has_losses = any([len(values) > 0 for values in self.losses.values()])
        if has_losses:
            loss_types_with_values = [loss_type for loss_type, values in self.losses.items() if len(values) > 0]
            num_loss_types = len(loss_types_with_values)
            
            if num_loss_types > 0:
                plt.figure(figsize=(12, 8))
                for i, loss_type in enumerate(loss_types_with_values):
                    values = self.losses[loss_type]
                    plt.subplot(num_loss_types, 1, i+1)
                    plt.plot(values, label=f'{loss_type.capitalize()} Loss')
                    plt.xlabel('Update Step')
                    plt.ylabel(f'{loss_type.capitalize()} Loss')
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, f"{filename_base}_losses.png"))
                plt.close()
        
        if self.exploration_values:
            plt.figure(figsize=(10, 6))
            plt.plot(self.exploration_values, label='Exploration Rate')
            plt.xlabel('Update Step')
            plt.ylabel('Exploration Rate')
            plt.title(f'{self.agent_name} Exploration Rate')
            plt.legend()
            plt.savefig(os.path.join(self.save_dir, f"{filename_base}_exploration.png"))
            plt.close()
        
        has_param_stats = any([len(values) > 0 for values in self.param_stats.values()])
        if has_param_stats:
            param_stats_with_values = [stat_type for stat_type, values in self.param_stats.items() if len(values) > 0]
            num_param_stats = len(param_stats_with_values)
            
            if num_param_stats > 0:
                plt.figure(figsize=(12, 10))
                for i, stat_type in enumerate(param_stats_with_values):
                    values = self.param_stats[stat_type]
                    plt.subplot(num_param_stats, 1, i+1)
                    plt.plot(values, label=f'{stat_type.replace("_", " ").capitalize()}')
                    plt.xlabel('Update Step')
                    plt.ylabel(f'{stat_type.replace("_", " ").capitalize()}')
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, f"{filename_base}_param_stats.png"))
                plt.close() 