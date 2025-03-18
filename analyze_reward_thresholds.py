import os
import pandas as pd
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt

def find_first_reward_threshold(csv_file, reward_thresholds):
    
    try:
        df = pd.read_csv(csv_file)
        
        results = {threshold: None for threshold in reward_thresholds}
        
        sorted_thresholds = sorted(reward_thresholds)
        
        # For each threshold, find the first episode where reward >= threshold
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
    # Extract the directory name which contains the method name
    dir_name = os.path.basename(os.path.dirname(file_path))
    
    # Extract the method name from the directory name
    if '_run_' in dir_name:
        method_name = dir_name.split('_run_')[0]
    else:
        method_name = dir_name
    
    return method_name

def find_latest_episode_file(method_dir):
    episode_files = glob.glob(os.path.join(method_dir, "*episodes.csv"))
    
    if not episode_files:
        return None
    
    # Sort files by episode number if available
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

def plot_reward_progression(method_files, window_size=100, figsize=(12, 8)):
    
    plt.figure(figsize=figsize)
    
    for method, file_path in method_files.items():
        try:
            df = pd.read_csv(file_path)
            
            # Calculate moving average if it doesn't exist
            if 'reward_moving_avg' not in df.columns or df['reward_moving_avg'].isna().all():
                df['reward_moving_avg'] = df['reward'].rolling(window=window_size, min_periods=1).mean()
            
            # Plot the moving average
            plt.plot(df['episode'], df['reward_moving_avg'], label=method)
            
        except Exception as e:
            print(f"Error plotting {method}: {e}")
    
    plt.title(f'Reward Progression (Moving Avg Window: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel(f'Reward (Moving Avg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_progression.png')
    print(f"Reward progression plot saved to reward_progression.png")

def plot_threshold_comparison(results, thresholds, figsize=(12, 8)):
    
    methods = [r['Method'] for r in results]
    
    plt.figure(figsize=figsize)
    
    bar_width = 0.8 / len(thresholds)
    
    r = np.arange(len(methods))
    
    for i, threshold in enumerate(thresholds):
        # Extract episodes for this threshold
        episodes = [r.get(f'Threshold_{threshold}') for r in results]
        
        # Convert None to NaN for plotting
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
    parser.add_argument('--csv_files', type=str, nargs='+',
                        help='Specific CSV files to analyze (overrides automatic file finding)')
    parser.add_argument('--method_names', type=str, nargs='+',
                        help='Method names corresponding to the CSV files (must match number of CSV files)')
    
    args = parser.parse_args()
    
    results = []
    method_files = {}
    
    if args.csv_files:
        if args.method_names and len(args.method_names) == len(args.csv_files):
            method_names = args.method_names
        else:
            # If method names are not provided or don't match, generate default names
            method_names = [f"Method_{i+1}" for i in range(len(args.csv_files))]
            if args.method_names:
                print("Warning: Number of method names doesn't match number of CSV files. Using default names.")
            else:
                print("No method names provided. Using default names.")
        
        # Process each CSV file
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
    else:
        # Use the original automatic file finding logic
        method_dirs = [d for d in glob.glob(os.path.join(args.metrics_dir, '*')) 
                      if os.path.isdir(d) and not d.startswith('.')]
        
        for method_dir in method_dirs:
            method_name = os.path.basename(method_dir)
            print(f"Processing method: {method_name}")
            
            latest_episode_file = find_latest_episode_file(method_dir)
            
            if latest_episode_file:
                print(f"  Using file: {os.path.basename(latest_episode_file)}")
                
                method_files[method_name] = latest_episode_file
                
                threshold_results = find_first_reward_threshold(latest_episode_file, args.reward_thresholds)
                
                result = {
                    'Method': method_name,
                    'File': os.path.basename(latest_episode_file)
                }
                
                for threshold, episode in threshold_results.items():
                    result[f'Threshold_{threshold}'] = episode
                    if episode is not None:
                        print(f"  First episode reaching reward {threshold}: {episode}")
                    else:
                        print(f"  Reward threshold {threshold} not reached")
                
                results.append(result)
            else:
                print(f"  No episode files found")
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")
        
        if not args.skip_plots:
            plot_reward_progression(method_files, window_size=args.window_size)
            
            plot_threshold_comparison(results, args.reward_thresholds)
    else:
        print("\nNo methods found or no results to report")

if __name__ == "__main__":
    main() 