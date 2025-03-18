import os
import pandas as pd
import glob
import argparse
import matplotlib.pyplot as plt

def find_first_reward_threshold(csv_file, reward_threshold):
    
    try:
        df = pd.read_csv(csv_file)
        
        # Find the first episode where reward >= threshold
        threshold_episodes = df[df['reward'] >= reward_threshold]
        
        if threshold_episodes.empty:
            return None
        
        # Get the first episode number
        first_episode = threshold_episodes.iloc[0]['episode']
        return first_episode
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None

def get_method_name(file_path):
    dir_name = os.path.basename(os.path.dirname(file_path))
    
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
                # Extract episode number from filename
                episode_str = filename.split('episode_')[1].split('_')[0]
                return int(episode_str)
            except (IndexError, ValueError):
                return 0
        return 0
    
    episode_files.sort(key=get_episode_number, reverse=True)
    return episode_files[0]

def main():
    parser = argparse.ArgumentParser(description='Find first episode reaching reward threshold')
    parser.add_argument('--reward_threshold', type=float, default=200.0,
                        help='Reward threshold to search for (default: 200.0)')
    parser.add_argument('--metrics_dir', type=str, default='metrics',
                        help='Directory containing metrics subdirectories (default: metrics)')
    parser.add_argument('--output_csv', type=str, default='reward_threshold_results.csv',
                        help='Output CSV file name (default: reward_threshold_results.csv)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate a bar plot of the results')
    parser.add_argument('--csv_files', type=str, nargs='+',
                        help='Specific CSV files to analyze (overrides automatic file finding)')
    parser.add_argument('--method_names', type=str, nargs='+',
                        help='Method names corresponding to the CSV files (must match number of CSV files)')
    
    args = parser.parse_args()
    
    results = []
    
    # If specific CSV files are provided, use them
    if args.csv_files:
        # Check if method names are provided and match the number of CSV files
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
                
                # Find the first episode reaching the threshold
                first_episode = find_first_reward_threshold(csv_file, args.reward_threshold)
                
                if first_episode is not None:
                    print(f"  First episode reaching reward {args.reward_threshold}: {first_episode}")
                    results.append({
                        'Method': method_name,
                        'First Episode': first_episode,
                        'Reward Threshold': args.reward_threshold,
                        'File': os.path.basename(csv_file)
                    })
                else:
                    print(f"  Reward threshold {args.reward_threshold} not reached")
            else:
                print(f"  File not found: {csv_file}")
    else:
        # Use the original automatic file finding logic
        method_dirs = [d for d in glob.glob(os.path.join(args.metrics_dir, '*')) 
                      if os.path.isdir(d) and not d.startswith('.')]
        
        for method_dir in method_dirs:
            method_name = os.path.basename(method_dir)
            print(f"Processing method: {method_name}")
            
            # Find the latest episode file
            latest_episode_file = find_latest_episode_file(method_dir)
            
            if latest_episode_file:
                print(f"  Using file: {os.path.basename(latest_episode_file)}")
                
                # Find the first episode reaching the threshold
                first_episode = find_first_reward_threshold(latest_episode_file, args.reward_threshold)
                
                if first_episode is not None:
                    print(f"  First episode reaching reward {args.reward_threshold}: {first_episode}")
                    results.append({
                        'Method': method_name,
                        'First Episode': first_episode,
                        'Reward Threshold': args.reward_threshold,
                        'File': os.path.basename(latest_episode_file)
                    })
                else:
                    print(f"  Reward threshold {args.reward_threshold} not reached")
            else:
                print(f"  No episode files found")
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")
        
        if args.plot:
            plt.figure(figsize=(10, 6))
            plt.bar(results_df['Method'], results_df['First Episode'])
            plt.title(f'First Episode Reaching Reward Threshold {args.reward_threshold}')
            plt.xlabel('Method')
            plt.ylabel('Episode Number')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('reward_threshold_plot.png')
            print(f"Plot saved to reward_threshold_plot.png")
    else:
        print("\nNo methods reached the specified reward threshold")

if __name__ == "__main__":
    main() 