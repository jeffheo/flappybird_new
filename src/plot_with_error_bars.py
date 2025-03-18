import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

def align_episodes_data(dataframes, x_col='episode'):
    """
    Aligns episode data from multiple runs to common x-axis points.
    
    Args:
        dataframes: List of pandas dataframes containing episode data
        x_col: Name of the column to use as x-axis
    Returns:
        Dictionary with common x values and aligned y values for each metric
    """
    all_x = set()
    for df in dataframes:
        all_x.update(df[x_col].values)
    common_x = sorted(list(all_x))
    
    metrics = []
    for df in dataframes:
        metrics.extend([col for col in df.columns if col != x_col])
    metrics = list(set(metrics))
    
    aligned_data = {
        'x': common_x,
        'metrics': {}
    }
    
    for metric in metrics:
        aligned_data['metrics'][metric] = []
    
    for df in dataframes:
        df_min_x = df[x_col].min()
        df_max_x = df[x_col].max()
        
        for metric in metrics:
            if metric in df.columns:
                interpolated = np.full(len(common_x), np.nan)
                valid_indices = [i for i, x in enumerate(common_x) if df_min_x <= x <= df_max_x]
                x_values_in_range = [common_x[i] for i in valid_indices]
                if x_values_in_range:
                    interp_values = np.interp(
                        x_values_in_range,
                        df[x_col].values,
                        df[metric].values
                    )
                for idx, value in zip(valid_indices, interp_values):
                    interpolated[idx] = value
                aligned_data['metrics'][metric].append(interpolated)
    return aligned_data

def calculate_statistics(aligned_data):
    """
    Calculate mean and standard deviation for each metric.
    
    Args:
        aligned_data: Dictionary with aligned data
        
    Returns:
        Dictionary with mean and std for each metric and count of valid data points
    """
    stats = {'x': aligned_data['x']}
    
    for metric, values_list in aligned_data['metrics'].items():
        values_array = np.array(values_list)
        mean = np.nanmean(values_array, axis=0)
        std = np.nanstd(values_array, axis=0)
        valid_count = np.sum(~np.isnan(values_array), axis=0)
        stats[metric] = {
            'mean': mean,
            'std': std,
            'valid_count': valid_count
        }
    return stats


def generate_plots_with_error_bars(csv_file_paths, output_dir=None, window_size=100, 
                                   agent_name=None, dpi=300, rewards_only=False):
    """
    Generate plots with error bars from multiple runs.
    
    Args:
        csv_file_paths (list): List of paths to episode CSV files
        output_dir (str, optional): Directory to save the plots. If None, uses the current directory.
        window_size (int): Window size for moving averages
        agent_name (str, optional): Name of the agent for plot titles
        dpi (int): DPI for saved plots
        rewards_only (bool): If True, only generate the rewards plot
    """
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    if agent_name is None:
        base_filename = os.path.splitext(os.path.basename(csv_file_paths[0]))[0].replace("_episodes", "")
        parts = base_filename.split('_')
        if len(parts) >= 1:
            agent_name = parts[0]
        else:
            agent_name = "Agent"
    
    episode_dfs = []
    
    for path in csv_file_paths:
        try:
            episode_df = pd.read_csv(path)
            
            if 'reward_moving_avg' not in episode_df.columns and len(episode_df) >= window_size:
                episode_df['reward_moving_avg'] = episode_df['reward'].rolling(window=window_size).mean()
            episode_dfs.append(episode_df)
        except Exception as e:
            print(f"Error loading data from {path}: {e}")
    
    if not episode_dfs:
        print("No valid episode data found")
        return
    
    aligned_episodes = align_episodes_data(episode_dfs, x_col='episode')
    episode_stats = calculate_statistics(aligned_episodes)
    
    plt.figure(figsize=(10, 6))
    
    x = episode_stats['x']
    
    if 'reward_moving_avg' in episode_stats:
        moving_avg_mean = episode_stats['reward_moving_avg']['mean']
        moving_avg_std = episode_stats['reward_moving_avg']['std']
        valid_count = episode_stats['reward_moving_avg']['valid_count']
        
        plt.plot(x, moving_avg_mean, color='red', linewidth=2, 
                 label=f'{window_size}-episode Moving Average')
        
        multi_run_mask = valid_count > 1
        
        if np.any(multi_run_mask):
            idx_with_variance = np.where(multi_run_mask)[0]
            
            if len(idx_with_variance) > 0:
                x_with_variance = np.array([x[i] for i in idx_with_variance])
                mean_with_variance = np.array([moving_avg_mean[i] for i in idx_with_variance])
                std_with_variance = np.array([moving_avg_std[i] for i in idx_with_variance])
                
                sort_idx = np.argsort(x_with_variance)
                x_with_variance = x_with_variance[sort_idx]
                mean_with_variance = mean_with_variance[sort_idx]
                std_with_variance = std_with_variance[sort_idx]
                
                plt.fill_between(x_with_variance, 
                                mean_with_variance - std_with_variance, 
                                mean_with_variance + std_with_variance, 
                                color='red', alpha=0.3, 
                                label='Standard Deviation')
    
    plt.title(f'{agent_name} Training Rewards - Moving Average (n={len(episode_dfs)} runs)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    
    rewards_path = os.path.join(output_dir, f"{agent_name}_rewards_moving_avg.png")
    plt.savefig(rewards_path, dpi=dpi)
    plt.close()
    print(f"Saved rewards moving average plot to {rewards_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate plots with error bars from multiple run CSVs')
    parser.add_argument('--agent', type=str, help='Agent name pattern to match (e.g., DQN, PPO)')
    parser.add_argument('--metrics_dir', type=str, default='metrics', help='Directory containing metrics files')
    parser.add_argument('--output_dir', type=str, help='Directory to save plots (default: final_plots)')
    parser.add_argument('--window_size', type=int, default=100, help='Window size for moving averages')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved plots')
    parser.add_argument('--rewards_only', action='store_true', help='Only generate the rewards plot')
    parser.add_argument('--files', nargs='+', help='Specific CSV files to process (instead of searching by agent)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = 'final_plots'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.files:
        csv_files = args.files
        agent_name = args.agent
        if agent_name is None:
            filename = os.path.basename(csv_files[0])
            parts = filename.split('_')
            if len(parts) >= 1:
                agent_name = parts[0]
            else:
                agent_name = "Agent"
        
        generate_plots_with_error_bars(
            csv_file_paths=csv_files,
            output_dir=args.output_dir,
            window_size=args.window_size,
            agent_name=agent_name,
            dpi=args.dpi,
            rewards_only=args.rewards_only,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 