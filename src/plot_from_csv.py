import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from glob import glob


def generate_plots_from_csv(episodes_csv_path, training_csv_path=None, output_dir=None, 
                           window_size=100, agent_name=None, dpi=300, rewards_only=False):
    
    # Extract base filename for output
    base_filename = os.path.splitext(os.path.basename(episodes_csv_path))[0].replace("_episodes", "")
    
    # If agent_name not provided, extract from filename
    if agent_name is None:
        parts = base_filename.split('_')
        if len(parts) >= 1:
            agent_name = parts[0]
        else:
            agent_name = "Agent"
    
    if output_dir is None:
        output_dir = os.path.dirname(episodes_csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        episode_df = pd.read_csv(episodes_csv_path)
        print(f"Loaded episode metrics from {episodes_csv_path}")
    except Exception as e:
        print(f"Error loading episode metrics: {e}")
        return
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episode_df['episode'], episode_df['reward'], alpha=0.3, color='blue', label='Rewards')
    
    # Plot moving average if available in CSV, otherwise calculate it
    if 'reward_moving_avg' in episode_df.columns:
        plt.plot(episode_df['episode'], episode_df['reward_moving_avg'], color='red', 
                 label=f'Moving Average')
    elif len(episode_df) >= window_size:
        moving_avg = episode_df['reward'].rolling(window=window_size).mean()
        plt.plot(episode_df['episode'], moving_avg, color='red', 
                 label=f'{window_size}-episode Moving Average')
    
    plt.title(f'{agent_name} Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    
    rewards_path = os.path.join(output_dir, f"{base_filename}_rewards.png")
    plt.savefig(rewards_path, dpi=dpi)
    plt.close()
    print(f"Saved rewards plot to {rewards_path}")
    
    # Plot episode length (only if rewards_only is False)
    if not rewards_only:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_df['episode'], episode_df['episode_length'], alpha=0.3, color='green', label='Episode Length')
        
        # Plot moving average if available in CSV, otherwise calculate it
        if 'length_moving_avg' in episode_df.columns:
            plt.plot(episode_df['episode'], episode_df['length_moving_avg'], color='orange', 
                     label=f'Moving Average')
        elif len(episode_df) >= window_size:
            moving_avg = episode_df['episode_length'].rolling(window=window_size).mean()
            plt.plot(episode_df['episode'], moving_avg, color='orange', 
                     label=f'{window_size}-episode Moving Average')
        
        plt.title(f'{agent_name} Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()
        plt.tight_layout()
        
        lengths_path = os.path.join(output_dir, f"{base_filename}_lengths.png")
        plt.savefig(lengths_path, dpi=dpi)
        plt.close()
        print(f"Saved episode lengths plot to {lengths_path}")
    
    if training_csv_path and os.path.exists(training_csv_path) and not rewards_only:
        try:
            training_df = pd.read_csv(training_csv_path)
            print(f"Loaded training metrics from {training_csv_path}")
        except Exception as e:
            print(f"Error loading training metrics: {e}")
            return
        
        loss_columns = [col for col in training_df.columns if col.endswith('_loss')]
        if loss_columns:
            plt.figure(figsize=(12, 8))
            for i, loss_col in enumerate(loss_columns):
                plt.subplot(len(loss_columns), 1, i+1)
                plt.plot(training_df['update_step'], training_df[loss_col], 
                         label=loss_col.replace('_loss', '').capitalize())
                plt.xlabel('Update Step')
                plt.ylabel(f'{loss_col.replace("_loss", "").capitalize()} Loss')
                plt.legend()
            
            plt.tight_layout()
            losses_path = os.path.join(output_dir, f"{base_filename}_losses.png")
            plt.savefig(losses_path, dpi=dpi)
            plt.close()
            print(f"Saved losses plot to {losses_path}")
        
        # Plot exploration if available
        if 'exploration' in training_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(training_df['update_step'], training_df['exploration'], label='Exploration Rate')
            plt.xlabel('Update Step')
            plt.ylabel('Exploration Rate')
            plt.title(f'{agent_name} Exploration Rate')
            plt.legend()
            exploration_path = os.path.join(output_dir, f"{base_filename}_exploration.png")
            plt.savefig(exploration_path, dpi=dpi)
            plt.close()
            print(f"Saved exploration plot to {exploration_path}")
        
        # Plot parameter statistics
        param_columns = [col for col in training_df.columns 
                         if col not in ['update_step'] + loss_columns + ['exploration']]
        
        if param_columns:
            plt.figure(figsize=(12, 10))
            for i, param_col in enumerate(param_columns):
                plt.subplot(len(param_columns), 1, i+1)
                plt.plot(training_df['update_step'], training_df[param_col], 
                         label=param_col.replace('_', ' ').capitalize())
                plt.xlabel('Update Step')
                plt.ylabel(param_col.replace('_', ' ').capitalize())
                plt.legend()
            
            plt.tight_layout()
            params_path = os.path.join(output_dir, f"{base_filename}_param_stats.png")
            plt.savefig(params_path, dpi=dpi)
            plt.close()
            print(f"Saved parameter statistics plot to {params_path}")


def find_matching_training_csv(episodes_csv_path):
    base_path = episodes_csv_path.replace('_episodes.csv', '')
    training_path = f"{base_path}_training.csv"
    if os.path.exists(training_path):
        return training_path
    return None


def main():
    parser = argparse.ArgumentParser(description='Generate plots from saved metrics CSV files')
    parser.add_argument('--episodes_csv', type=str, help='Path to episodes CSV file')
    parser.add_argument('--training_csv', type=str, help='Path to training CSV file (optional)')
    parser.add_argument('--output_dir', type=str, help='Directory to save plots (default: same as CSV)')
    parser.add_argument('--window_size', type=int, default=100, help='Window size for moving averages')
    parser.add_argument('--agent_name', type=str, help='Agent name for plot titles')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved plots')
    parser.add_argument('--all', action='store_true', help='Process all CSV files in the metrics directory')
    parser.add_argument('--metrics_dir', type=str, default='metrics', help='Directory containing metrics files')
    parser.add_argument('--rewards_only', action='store_true', help='Only generate the rewards plot')
    
    args = parser.parse_args()
    
    if args.all:
        episodes_files = glob(os.path.join(args.metrics_dir, '*_episodes.csv'))
        if not episodes_files:
            print(f"No episodes CSV files found in {args.metrics_dir}")
            return
        
        for episodes_file in episodes_files:
            training_file = find_matching_training_csv(episodes_file)
            generate_plots_from_csv(
                episodes_csv_path=episodes_file,
                training_csv_path=training_file,
                output_dir=args.output_dir,
                window_size=args.window_size,
                agent_name=args.agent_name,
                dpi=args.dpi,
                rewards_only=args.rewards_only
            )
    elif args.episodes_csv:
        training_csv = args.training_csv or find_matching_training_csv(args.episodes_csv)
        generate_plots_from_csv(
            episodes_csv_path=args.episodes_csv,
            training_csv_path=training_csv,
            output_dir=args.output_dir,
            window_size=args.window_size,
            agent_name=args.agent_name,
            dpi=args.dpi,
            rewards_only=args.rewards_only
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 