import os
import pandas as pd
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt

def find_first_reward_threshold(csv_file, reward_thresholds):
    """
    Find the first episode where each reward threshold is reached.
    
    Args:
        csv_file: Path to the episodes CSV file
        reward_thresholds: List of reward thresholds to check for
        
    Returns:
        dict: Mapping of threshold -> first episode number
              (None if threshold is never reached)
    """
    try:
        df = pd.read_csv(csv_file)
        
        results = {threshold: None for threshold in reward_thresholds}
        
        sorted_thresholds = sorted(reward_thresholds)
        
        for threshold in sorted_thresholds:
            threshold_episodes = df[df['reward'] >= threshold]
            
            if not threshold_episodes.empty:
                first_episode = threshold_episodes.iloc[0]['episode']
                results[threshold] = first_episode
        
        return results
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return {threshold: None for threshold in reward_thresholds}

def get_method_name(file_path):
    """Extract the method name from the file path"""
    dir_name = os.path.basename(os.path.dirname(file_path))
    
    if '_run_' in dir_name:
        method_name = dir_name.split('_run_')[0]
    else:
        method_name = dir_name
    
    return method_name

def find_latest_episode_file(method_dir):
    """Find the latest (highest episode number) episodes CSV file in a method directory"""
    episode_files = glob.glob(os.path.join(method_dir, "*episodes.csv"))
    
    if not episode_files:
        return None
    
    def get_episode_number(file_path):
        filename = os.path.basename(file_path)
        if 'episode_' in filename:
            try:
                episode_str = filename.split('episode_')[1].split('_')[0]
                return int(episode_str)
            except (IndexError, ValueError):
                return 0
        return 0
    
    episode_files.sort(key=get_episode_number, reverse=True)
    return episode_files[0]

def plot_reward_progression(method_files, window_size=100, figsize=(12, 8), use_subplots=False):
    """
    Plot the reward progression over episodes for multiple methods.
    
    Args:
        method_files: Dict mapping method names to episode CSV files
        window_size: Size of the moving average window
        figsize: Figure size for the plot
        use_subplots: If True, plot each method in a separate subplot
    """
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    line_styles = ['-', '--', '-.', ':']
    
    if use_subplots:
        num_methods = len(method_files)
        fig, axes = plt.subplots(num_methods, 1, figsize=(figsize[0], figsize[1] * num_methods // 2), sharex=True)
        
        for i, (method, file_path) in enumerate(method_files.items()):
            try:
                df = pd.read_csv(file_path)
                
                if 'reward_moving_avg' not in df.columns or df['reward_moving_avg'].isna().all():
                    df['reward_moving_avg'] = df['reward'].rolling(window=window_size, min_periods=1).mean()
                
                ax = axes[i] if num_methods > 1 else axes
                
                color_idx = i % len(colors)
                style_idx = (i // len(colors)) % len(line_styles)
                ax.plot(df['episode'], df['reward_moving_avg'], 
                        color=colors[color_idx], 
                        linestyle=line_styles[style_idx],
                        label=method)
                
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
                ax.set_ylabel(f'Reward (Moving Avg)')
                ax.set_title(method)
                
            except Exception as e:
                print(f"Error plotting {method}: {e}")
        
        plt.xlabel('Episode')
        fig.suptitle(f'Reward Progression (Moving Avg Window: {window_size})')
        plt.tight_layout()
        
    else:
        plt.figure(figsize=figsize)
        
        for i, (method, file_path) in enumerate(method_files.items()):
            try:
                df = pd.read_csv(file_path)
                
                if 'reward_moving_avg' not in df.columns or df['reward_moving_avg'].isna().all():
                    df['reward_moving_avg'] = df['reward'].rolling(window=window_size, min_periods=1).mean()
                
                color_idx = i % len(colors)
                style_idx = (i // len(colors)) % len(line_styles)
                plt.plot(df['episode'], df['reward_moving_avg'], 
                        color=colors[color_idx], 
                        linestyle=line_styles[style_idx],
                        linewidth=2,
                        label=method)
                
            except Exception as e:
                print(f"Error plotting {method}: {e}")
        
        plt.title(f'Reward Progression (Moving Avg Window: {window_size})')
        plt.xlabel('Episode')
        plt.ylabel(f'Reward (Moving Avg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    plt.savefig('reward_progression.png', dpi=300)
    print(f"Reward progression plot saved to reward_progression.png")
    
    plot_grouped_by_range(method_files, window_size, figsize)

def plot_grouped_by_range(method_files, window_size=100, figsize=(12, 8)):
    """
    Group methods by their value ranges and plot them separately.
    This helps when some methods have much higher values than others.
    """
    method_max_values = {}
    all_data = {}
    
    for method, file_path in method_files.items():
        try:
            df = pd.read_csv(file_path)
            
            if 'reward_moving_avg' not in df.columns or df['reward_moving_avg'].isna().all():
                df['reward_moving_avg'] = df['reward'].rolling(window=window_size, min_periods=1).mean()
            
            method_max_values[method] = df['reward_moving_avg'].max()
            all_data[method] = df
            
        except Exception as e:
            print(f"Error analyzing {method}: {e}")
    
    if not method_max_values:
        return  # No data to plot
    
    thresholds = [0, 500, 1000, float('inf')]
    range_labels = [f"{thresholds[i]} - {thresholds[i+1]}" for i in range(len(thresholds)-1)]
    
    methods_by_range = {label: [] for label in range_labels}
    
    for method, max_val in method_max_values.items():
        for i, (lower, upper) in enumerate(zip(thresholds[:-1], thresholds[1:])):
            if lower <= max_val < upper:
                methods_by_range[range_labels[i]].append(method)
                break
    
    methods_by_range = {k: v for k, v in methods_by_range.items() if v}
    
    if not methods_by_range:
        return  # No groups to plot
    
    num_ranges = len(methods_by_range)
    fig, axes = plt.subplots(num_ranges, 1, figsize=(figsize[0], figsize[1] * num_ranges // 2), sharex=True)
    
    if num_ranges == 1:
        axes = [axes]
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    line_styles = ['-', '--', '-.', ':']
    
    for i, (range_label, methods) in enumerate(methods_by_range.items()):
        ax = axes[i]
        
        for j, method in enumerate(methods):
            df = all_data[method]
            
            color_idx = j % len(colors)
            style_idx = (j // len(colors)) % len(line_styles)
            ax.plot(df['episode'], df['reward_moving_avg'], 
                    color=colors[color_idx], 
                    linestyle=line_styles[style_idx],
                    linewidth=2,
                    label=method)
        
        ax.set_title(f'Reward Range: {range_label}')
        ax.set_ylabel('Reward (Moving Avg)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.xlabel('Episode')
    fig.suptitle(f'Reward Progression Grouped by Range (Window: {window_size})')
    plt.tight_layout()
    
    plt.savefig('reward_progression_by_range.png', dpi=300)
    print(f"Range-grouped reward progression plot saved to reward_progression_by_range.png")

def plot_threshold_comparison(results, thresholds, figsize=(12, 8)):
    """
    Create a grouped bar chart comparing when each method reached each threshold.
    
    Args:
        results: List of dictionaries with results
        thresholds: List of reward thresholds
        figsize: Figure size for the plot
    """
    methods = [r['Method'] for r in results]
    
    plt.figure(figsize=figsize)
    
    bar_width = 0.8 / len(thresholds)
    
    r = np.arange(len(methods))
    
    for i, threshold in enumerate(thresholds):
        episodes = [r.get(f'Threshold_{threshold}') for r in results]
        episodes = [float('nan') if e is None else e for e in episodes]
        plt.bar(r + i * bar_width, episodes, width=bar_width, 
                label=f'Reward ≥ {threshold}', alpha=0.7)
    
    plt.xlabel('Method')
    plt.ylabel('First Episode Reaching Threshold')
    plt.title('Episode When Reward Threshold Was First Reached')
    plt.xticks(r + bar_width * (len(thresholds) - 1) / 2, methods, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('threshold_comparison.png')
    print(f"Threshold comparison plot saved to threshold_comparison.png")

def main():
    parser = argparse.ArgumentParser(description='Analyze reward thresholds in reinforcement learning metrics')
    parser.add_argument('--reward_thresholds', type=float, nargs='+', default=[100.0, 200.0, 300.0, 400.0],
                        help='Reward thresholds to search for (default: 100.0 200.0 300.0 400.0)')
    parser.add_argument('--metrics_dir', type=str, default='metrics',
                        help='Directory containing metrics subdirectories (default: metrics)')
    parser.add_argument('--output_csv', type=str, default='reward_threshold_results.csv',
                        help='Output CSV file name (default: reward_threshold_results.csv)')
    parser.add_argument('--window_size', type=int, default=100,
                        help='Window size for moving average in progression plot (default: 100)')
    parser.add_argument('--skip_plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--use_subplots', action='store_true',
                        help='Plot each method in a separate subplot')
    parser.add_argument('--csv_files', type=str, nargs='+',
                        help='Specific CSV files to analyze (overrides automatic file finding)')
    parser.add_argument('--method_names', type=str, nargs='+',
                        help='Method names corresponding to the CSV files (must match number of CSV files)')
    
    args = parser.parse_args()
    
    results = []
    method_files = {}
    
    if args.method_names and len(args.method_names) == len(args.csv_files):
        method_names = args.method_names
    else:
        method_names = [f"Method_{i+1}" for i in range(len(args.csv_files))]
        if args.method_names:
            print("Warning: Number of method names doesn't match number of CSV files. Using default names.")
        else:
            print("No method names provided. Using default names.")
    
    for i, csv_file in enumerate(args.csv_files):
        method_name = method_names[i]
        print(f"Processing method: {method_name}")
        
        if os.path.exists(csv_file):
            print(f"  Using file: {os.path.basename(csv_file)}")
            
            method_files[method_name] = csv_file
            threshold_results = find_first_reward_threshold(csv_file, args.reward_thresholds)
            result = {
                'Method': method_name,
                'File': os.path.basename(csv_file)
            }
            
            for threshold, episode in threshold_results.items():
                result[f'Threshold_{threshold}'] = episode
                if episode is not None:
                    print(f"  First episode reaching reward {threshold}: {episode}")
                else:
                    print(f"  Reward threshold {threshold} not reached")
            
            results.append(result)
        else:
            print(f"  File not found: {csv_file}")
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")
        
        if not args.skip_plots:
            plot_reward_progression(method_files, window_size=args.window_size, use_subplots=args.use_subplots)
            plot_threshold_comparison(results, args.reward_thresholds)
    else:
        print("\nNo methods found or no results to report")

if __name__ == "__main__":
    main() 