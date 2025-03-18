import numpy as np
import matplotlib.pyplot as plt
import flappy_bird_gym
from src.trainer import Trainer
from src.dqn.agent_unified import AgentUnified
from src.dqn.config_unified import UnifiedAgentConfig
import torch
import argparse
import os
from datetime import datetime

def create_run_directory(feature_extractor, suffix=""):
    """Create a unique directory for this training run"""
    # Create a timestamp for unique identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"DQN_{feature_extractor}{suffix}_{timestamp}"
    
    # Create base metrics directory if it doesn't exist
    base_dir = "metrics"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create run-specific directory
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create models directory within run directory
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Create checkpoints directory within run directory
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Create plots directory within run directory
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Training results will be saved to: {run_dir}")
    
    return run_dir, models_dir, checkpoints_dir, plots_dir, run_name

def plot_scores(scores, window_size=100, feature_extractor='resnet', suffix='', save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.3, color='blue', label='Raw Scores')
    
    # Plot moving average only if we have enough data
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(scores)), moving_avg, color='red', 
                label=f'{window_size}-episode Moving Average')
    else:
        print(f"Warning: Not enough episodes ({len(scores)}) to calculate {window_size}-episode moving average.")
        # Use a smaller window size if possible
        if len(scores) > 5:
            smaller_window = min(len(scores) // 2, 20)  # Use half the episodes or 20, whichever is smaller
            moving_avg = np.convolve(scores, np.ones(smaller_window)/smaller_window, mode='valid')
            plt.plot(range(smaller_window-1, len(scores)), moving_avg, color='red', 
                    label=f'{smaller_window}-episode Moving Average')
    
    plt.title(f'DQN Training Progress ({feature_extractor.capitalize()})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    
    # Save to the specified path or default location
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig(f'training_progress_{feature_extractor}{suffix}.png')
    
    plt.close()

def save_training_results(agent, trainer, run_dir, models_dir, plots_dir, feature_extractor, suffix, seed, final=False):
    """Save model, plot scores, and print statistics"""
    try:
        # Save the model
        if final:
            final_model_path = os.path.join(models_dir, f"DQN_{feature_extractor}{suffix}_final.pt")
            agent.save_model(final_model_path)
            print(f"Final model saved to: {final_model_path}", flush=True)
        
        # Plot and save the learning curve
        plot_path = os.path.join(run_dir, f"training_progress_{feature_extractor}{suffix}_final.png")
        plot_scores(trainer.scores, feature_extractor=feature_extractor, suffix=suffix, save_path=plot_path)
        print(f"Final plot saved to: {plot_path}", flush=True)
        
        # Print final statistics
        print(f"\nTraining {'completed' if final else 'interrupted'} after {trainer.last_episode} episodes", flush=True)
        print(f"Best score: {trainer.best_score}", flush=True)
        print(f"Final average score: {np.mean(trainer.scores_window):.2f}", flush=True)
        
        # Print metrics summary
        if hasattr(trainer, 'metrics') and trainer.metrics:
            stats = trainer.metrics.get_current_stats()
            print("\nMetrics Summary:", flush=True)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}", flush=True)
                else:
                    print(f"{key}: {value}", flush=True)
            
            # Save metrics explicitly
            try:
                trainer.metrics.save_metrics(additional_info={
                    "final": final, 
                    "interrupted": not final, 
                    "seed": seed,
                    "feature_extractor": feature_extractor
                }, generate_plots=True)
                print(f"Metrics saved successfully to {run_dir}", flush=True)
            except Exception as e:
                print(f"Error saving metrics: {e}", flush=True)
    except Exception as e:
        print(f"Error saving final results: {e}", flush=True)

def create_checkpoint_callback(agent, trainer, run_dir, checkpoints_dir, plots_dir, feature_extractor, suffix, seed, checkpoint_frequency=100):
    """
    Create a callback function for periodic checkpointing.
    
    Args:
        agent: The agent to save
        trainer: The trainer object
        run_dir: Directory to save metrics
        checkpoints_dir: Directory to save model checkpoints
        plots_dir: Directory to save plots
        feature_extractor: Name of the feature extractor
        suffix: Suffix for filenames
        seed: Random seed used for training
        checkpoint_frequency: How often to save checkpoints (in episodes)
        
    Returns:
        checkpoint_callback: Function to be called after each episode
    """
    def checkpoint_callback(trainer, episode):
        # Save checkpoint if it's time
        if episode % checkpoint_frequency == 0:
            try:
                checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_ep{episode}.pt")
                agent.save_model(checkpoint_path)
                print(f"Checkpoint saved at episode {episode}: {checkpoint_path}", flush=True)
                
                # Save current metrics
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    trainer.metrics.save_metrics(additional_info={
                        "checkpoint": True,
                        "episode": episode,
                        "seed": seed,
                        "feature_extractor": feature_extractor
                    }, generate_plots=False)
            except Exception as e:
                print(f"Failed to save checkpoint: {e}", flush=True)
    
    return checkpoint_callback

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DQN with various feature extractors on Flappy Bird')
    parser.add_argument('--suffix', type=str, default='', help='Suffix to append to output filenames')
    parser.add_argument('--feature_extractor', type=str, default='resnet', 
                        choices=['spatial_cnn', 'resnet', 'efficientnet'],
                        help='Feature extractor to use')
    parser.add_argument('--finetune', action='store_true', 
                        help='Whether to fine-tune the feature extractor')
    parser.add_argument('--episodes', type=int, default=100000, 
                        help='Number of episodes to train for')
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size')
    parser.add_argument('--hidden_size', type=str, default='256,128', 
                        help='Hidden layer sizes (comma-separated)')
    parser.add_argument('--frame_stack', action='store_true', default=False,
                        help='Whether to use frame stacking')
    parser.add_argument('--frame_stack_size', type=int, default=2,
                        help='Number of frames to stack')
    parser.add_argument('--target_size', type=str, default='84,84',
                        help='Target size for processed images (height,width)')
    parser.add_argument('--preprocess_method', type=str, default='enhanced',
                        choices=['basic', 'enhanced', 'context'],
                        help='Preprocessing method')
    parser.add_argument('--early_stop', type=int, default=5000,
                        help='Early stopping score threshold')
    parser.add_argument('--checkpoint_freq', type=int, default=100,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    args = parser.parse_args()
    
    # Create suffix with underscore if provided
    suffix = f"_{args.suffix}" if args.suffix else ""
    
    # Create run directories
    run_dir, models_dir, checkpoints_dir, plots_dir, run_name = create_run_directory(
        args.feature_extractor, suffix
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # Disable pygame audio to avoid ALSA errors
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

    # Initialize environment
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    
    # Set random seed for reproducibility
    seed = args.seed
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    hidden_size = tuple(map(int, args.hidden_size.split(',')))
    
    target_size = tuple(map(int, args.target_size.split(',')))
    
    if args.frame_stack:
        
        state_shape = (args.frame_stack_size, target_size[0], target_size[1])
    else:
        
        state_shape = (1, target_size[0], target_size[1])
    
    # Print preprocessing configuration
    print(f"Using {args.feature_extractor} feature extractor", flush=True)
    print(f"Preprocessing method: {args.preprocess_method}", flush=True)
    print(f"Target size: {target_size}", flush=True)
    print(f"Frame stacking: {args.frame_stack}", flush=True)
    if args.frame_stack:
        print(f"Frame stack size: {args.frame_stack_size}", flush=True)
    print(f"State shape: {state_shape}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Run ID: {run_name}", flush=True)

    # Model paths
    model_path = os.path.join(models_dir, f"DQN_{args.feature_extractor}{suffix}.pt")
    best_model_path = os.path.join(models_dir, f"DQN_{args.feature_extractor}{suffix}_best.pt")
    
    # Initialize agent with parameters
    params = UnifiedAgentConfig(
        state_size=state_shape,
        action_size=2,
        seed=seed,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gamma=0.99,
        tau=0.001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99995,  # Slower decay for more exploration
        model_dir=model_path,
        nb_hidden=hidden_size,  # Set hidden_size in the config
        feature_extractor=args.feature_extractor,
        finetune_features=args.finetune,
        use_frame_stack=args.frame_stack,
        frame_stack_size=args.frame_stack_size,
        target_size=target_size,
        preprocess_method=args.preprocess_method
    )

    # Create agent with the specified feature extractor
    agent = AgentUnified(
        **params.dict()
    )
    
    # Create checkpoint callback
    checkpoint_callback = create_checkpoint_callback(
        agent=agent,
        trainer=None,  # Will be set later
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        plots_dir=plots_dir,
        feature_extractor=args.feature_extractor,
        suffix=suffix,
        seed=seed,
        checkpoint_frequency=args.checkpoint_freq
    )
    
    # Training parameters
    trainer_args = {
        "n_episodes": args.episodes,
        "print_range": 100,
        "early_stop": args.early_stop,
        "max_timestep": 5000,  # Prevent very long episodes
        "verbose": True,
        "checkpoint_callback": checkpoint_callback  # Add our custom callback
    }

    print("Starting training...", flush=True)
    print(f"Checkpoints will be saved every {args.checkpoint_freq} episodes to {checkpoints_dir}", flush=True)
    # Create trainer and run training
    trainer = Trainer(agent=agent, env=env, **trainer_args)
    
    try:
        # Reset the agent's preprocessor before starting training
        agent.reset()
        
        # Run training
        trainer.run(
            logs_callback=agent.logs, 
            save_best_model=True, 
            output_path=best_model_path,
            run_dir=run_dir,
            agent_name=run_name
        )
        
        # Save final results
        save_training_results(
            agent=agent, 
            trainer=trainer, 
            run_dir=run_dir, 
            models_dir=models_dir, 
            plots_dir=plots_dir,
            feature_extractor=args.feature_extractor, 
            suffix=suffix, 
            seed=seed, 
            final=True
        )
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user", flush=True)
        # Save intermediate results
        save_training_results(
            agent=agent, 
            trainer=trainer, 
            run_dir=run_dir, 
            models_dir=models_dir, 
            plots_dir=plots_dir,
            feature_extractor=args.feature_extractor, 
            suffix=suffix, 
            seed=seed, 
            final=False
        )
    except Exception as e:
        print(f"\nError during training: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    main() 