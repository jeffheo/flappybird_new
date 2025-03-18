import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from glob import glob


def align_episodes_data(dataframes, x_col='episode', align_method='interpolate'):
    """
    Aligns episode data from multiple runs to common x-axis points.
    
    Args:
        dataframes: List of pandas dataframes containing episode data
        x_col: Name of the column to use as x-axis
        align_method: Method to align data ('interpolate' or 'bin')
        
    Returns:
        Dictionary with common x values and aligned y values for each metric
    """
    # Find global min and max x values
    global_min_x = min(df[x_col].min() for df in dataframes)
    global_max_x = max(df[x_col].max() for df in dataframes)
    
    # Create common x points
    if align_method == 'interpolate':
        # Get all unique x values across all dataframes
        all_x = set()
        for df in dataframes:
            all_x.update(df[x_col].values)
        common_x = sorted(list(all_x))
    else:  # bin method
        # Create evenly spaced bins
        num_bins = 100  # Adjust as needed
        common_x = np.linspace(global_min_x, global_max_x, num_bins)
    
    # Get all metrics (columns) except x_col
    metrics = []
    for df in dataframes:
        metrics.extend([col for col in df.columns if col != x_col])
    metrics = list(set(metrics))
    
    # Initialize result dictionary
    aligned_data = {
        'x': common_x,
        'metrics': {}
    }
    
    # Initialize arrays for each metric
    for metric in metrics:
        aligned_data['metrics'][metric] = []
    
    # Align each dataframe to common x points
    for df in dataframes:
        df_min_x = df[x_col].min()
        df_max_x = df[x_col].max()
        
        for metric in metrics:
            if metric in df.columns:
                if align_method == 'interpolate':
                    # Create array of NaNs for the full range
                    interpolated = np.full(len(common_x), np.nan)
                    
                    # Get indices where x values are within this dataframe's range
                    valid_indices = [i for i, x in enumerate(common_x) if df_min_x <= x <= df_max_x]
                    
                    # Only interpolate within the dataframe's range
                    x_values_in_range = [common_x[i] for i in valid_indices]
                    
                    if x_values_in_range:  # Check if there are any valid x values
                        interp_values = np.interp(
                            x_values_in_range,
                            df[x_col].values,
                            df[metric].values
                        )
                        
                        # Assign interpolated values
                        for idx, value in zip(valid_indices, interp_values):
                            interpolated[idx] = value
                    
                    aligned_data['metrics'][metric].append(interpolated)
                else:  # bin method
                    # Create bins and calculate mean for each bin
                    binned_values = np.full(len(common_x), np.nan)
                    bin_width = (global_max_x - global_min_x) / (len(common_x) - 1)
                    
                    for i, x in enumerate(common_x):
                        # Skip bins outside this dataframe's range
                        if x < df_min_x or x > df_max_x:
                            continue
                            
                        bin_min = x - bin_width/2
                        bin_max = x + bin_width/2
                        
                        # Handle edge cases
                        if i == 0:
                            bin_min = x
                        if i == len(common_x) - 1:
                            bin_max = x
                            
                        bin_data = df[(df[x_col] >= bin_min) & (df[x_col] <= bin_max)]
                        if not bin_data.empty:
                            binned_values[i] = bin_data[metric].mean()
                    
                    aligned_data['metrics'][metric].append(binned_values)
    
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
        # Convert to numpy array for easier calculations
        values_array = np.array(values_list)
        
        # Calculate mean and std, ignoring NaN values
        mean = np.nanmean(values_array, axis=0)
        std = np.nanstd(values_array, axis=0)
        
        # Count non-NaN values at each point
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
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    # If agent_name not provided, try to extract from filename
    if agent_name is None:
        # Try to extract agent name from first filename
        base_filename = os.path.splitext(os.path.basename(csv_file_paths[0]))[0].replace("_episodes", "")
        parts = base_filename.split('_')
        if len(parts) >= 1:
            agent_name = parts[0]
        else:
            agent_name = "Agent"
    
    # Load all dataframes
    episode_dfs = []
    
    for path in csv_file_paths:
        try:
            # Load episode data
            episode_df = pd.read_csv(path)
            
            # Calculate moving average if not present
            if 'reward_moving_avg' not in episode_df.columns and len(episode_df) >= window_size:
                episode_df['reward_moving_avg'] = episode_df['reward'].rolling(window=window_size).mean()
            
            episode_dfs.append(episode_df)
        except Exception as e:
            print(f"Error loading data from {path}: {e}")
    
    if not episode_dfs:
        print("No valid episode data found")
        return
    
    # Align episode data
    aligned_episodes = align_episodes_data(episode_dfs, x_col='episode')
    episode_stats = calculate_statistics(aligned_episodes)
    
    # Plot rewards with error bars - only moving average
    plt.figure(figsize=(10, 6))
    
    x = episode_stats['x']
    
    # Plot moving average with shaded error region if available
    if 'reward_moving_avg' in episode_stats:
        moving_avg_mean = episode_stats['reward_moving_avg']['mean']
        moving_avg_std = episode_stats['reward_moving_avg']['std']
        valid_count = episode_stats['reward_moving_avg']['valid_count']
        
        # Plot the mean line for the entire range
        plt.plot(x, moving_avg_mean, color='red', linewidth=2, 
                 label=f'{window_size}-episode Moving Average')
        
        # Create a mask for regions with multiple runs (where standard deviation is meaningful)
        multi_run_mask = valid_count > 1
        
        # Only show variance where we have multiple runs
        if np.any(multi_run_mask):
            # Instead of using boolean mask, collect indices where mask is True
            idx_with_variance = np.where(multi_run_mask)[0]
            
            if len(idx_with_variance) > 0:
                x_with_variance = np.array([x[i] for i in idx_with_variance])
                mean_with_variance = np.array([moving_avg_mean[i] for i in idx_with_variance])
                std_with_variance = np.array([moving_avg_std[i] for i in idx_with_variance])
                
                # Sort the arrays by x value to ensure proper plotting
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


def find_csv_files_by_pattern(agent_pattern, metrics_dir, seed_filter=None):
    """
    Find all episode CSV files matching the agent pattern.
    
    Args:
        agent_pattern (str): Pattern to match agent name (e.g., 'DQN', 'PPO')
        metrics_dir (str): Directory containing metrics files
        seed_filter (list, optional): List of seeds to include
        
    Returns:
        List of paths to episode CSV files
    """
    # Find all subdirectories matching the pattern
    agent_dirs = []
    for item in os.listdir(metrics_dir):
        item_path = os.path.join(metrics_dir, item)
        if os.path.isdir(item_path) and agent_pattern in item:
            agent_dirs.append(item_path)
    
    # Find all episode CSV files in those directories
    csv_files = []
    for agent_dir in agent_dirs:
        episode_files = glob(os.path.join(agent_dir, '*_episodes.csv'))
        
        # Apply seed filter if provided
        if seed_filter:
            filtered_files = []
            for file in episode_files:
                # Extract seed from filename if present
                filename = os.path.basename(file)
                if "_seed" in filename:
                    seed = filename.split("_seed")[1].split("_")[0]
                    if seed in seed_filter:
                        filtered_files.append(file)
                else:
                    # If no seed info in filename, keep the file
                    filtered_files.append(file)
            episode_files = filtered_files
        
        csv_files.extend(episode_files)
    
    return csv_files


def main():
    parser = argparse.ArgumentParser(description='Generate plots with error bars from multiple run CSVs')
    parser.add_argument('--agent', type=str, help='Agent name pattern to match (e.g., DQN, PPO)')
    parser.add_argument('--metrics_dir', type=str, default='metrics', help='Directory containing metrics files')
    parser.add_argument('--output_dir', type=str, help='Directory to save plots (default: final_plots)')
    parser.add_argument('--window_size', type=int, default=100, help='Window size for moving averages')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved plots')
    parser.add_argument('--rewards_only', action='store_true', help='Only generate the rewards plot')
    parser.add_argument('--files', nargs='+', help='Specific CSV files to process (instead of searching by agent)')
    parser.add_argument('--seeds', nargs='+', help='Specific seeds to include')
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = 'final_plots'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.files:
        # Use specific files provided
        csv_files = args.files
        
        # Extract agent name from first file if not specified
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
    elif args.agent:
        # Find CSV files matching the agent pattern
        csv_files = find_csv_files_by_pattern(
            agent_pattern=args.agent,
            metrics_dir=args.metrics_dir,
            seed_filter=args.seeds
        )
        
        if not csv_files:
            print(f"No CSV files found for agent pattern '{args.agent}'")
            return
        
        generate_plots_with_error_bars(
            csv_file_paths=csv_files,
            output_dir=args.output_dir,
            window_size=args.window_size,
            agent_name=args.agent,
            dpi=args.dpi,
            rewards_only=args.rewards_only,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 