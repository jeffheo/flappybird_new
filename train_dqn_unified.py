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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"DQN_{feature_extractor}{suffix}_{timestamp}"
    
    base_dir = "metrics"
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return run_dir, models_dir, checkpoints_dir, plots_dir, run_name

def plot_scores(scores, window_size=100, feature_extractor='resnet', suffix='', save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.3, color='blue', label='Raw Scores')
    assert len(scores) >= window_size, f"Not enough episodes ({len(scores)}) to calculate {window_size}-episode moving average."
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(scores)), moving_avg, color='red', 
            label=f'{window_size}-episode Moving Average')
    
    plt.title(f'DQN Training Progress ({feature_extractor.capitalize()})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig(f'training_progress_{feature_extractor}{suffix}.png')
    
    plt.close()

def save_training_results(agent, trainer, run_dir, models_dir, plots_dir, feature_extractor, suffix, seed, final=False):
    """Save model, plot scores, and print statistics"""
    try:
        if final:
            final_model_path = os.path.join(models_dir, f"DQN_{feature_extractor}{suffix}_final.pt")
            agent.save_model(final_model_path)
            print(f"Final model saved to: {final_model_path}", flush=True)
        
        plot_path = os.path.join(run_dir, f"training_progress_{feature_extractor}{suffix}_final.png")
        plot_scores(trainer.scores, feature_extractor=feature_extractor, suffix=suffix, save_path=plot_path)
        print(f"Final plot saved to: {plot_path}", flush=True)
        
        print(f"\nTraining {'completed' if final else 'interrupted'} after {trainer.last_episode} episodes", flush=True)
        print(f"Best score: {trainer.best_score}", flush=True)
        print(f"Final average score: {np.mean(trainer.scores_window):.2f}", flush=True)
        
        if hasattr(trainer, 'metrics') and trainer.metrics:
            stats = trainer.metrics.get_current_stats()
            print("\nMetrics Summary:", flush=True)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}", flush=True)
                else:
                    print(f"{key}: {value}", flush=True)
            
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
        if episode % checkpoint_frequency == 0:
            try:
                checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_ep{episode}.pt")
                agent.save_model(checkpoint_path)
                print(f"Checkpoint saved at episode {episode}: {checkpoint_path}", flush=True)
                
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
    parser = argparse.ArgumentParser(description='Train DQN with various feature extractors on Flappy Bird')
    parser.add_argument('--suffix', type=str, default='', help='Suffix to append to output filenames')
    parser.add_argument('--feature_extractor', type=str, default='resnet', 
                        choices=['spatial_cnn', 'resnet'],
                        help='Feature extractor to use')
    parser.add_argument('--finetune', action='store_true', 
                        help='Whether to fine-tune the feature extractor')
    parser.add_argument('--episodes', type=int, default=100000, 
                        help='Number of episodes to train for')
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--hidden_size', type=str, default='64,64', 
                        help='Hidden layer sizes (comma-separated)')
    parser.add_argument('--target_size', type=str, default='224,224',
                        help='Target size for processed images (height,width)')
    parser.add_argument('--early_stop', type=int, default=5000,
                        help='Early stopping score threshold')
    parser.add_argument('--checkpoint_freq', type=int, default=100,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    args = parser.parse_args()
    
    suffix = f"_{args.suffix}" if args.suffix else ""
    run_dir, models_dir, checkpoints_dir, plots_dir, run_name = create_run_directory(
        args.feature_extractor, suffix
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    
    seed = args.seed
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    hidden_size = tuple(map(int, args.hidden_size.split(',')))
    target_size = tuple(map(int, args.target_size.split(',')))
    state_shape = (1, target_size[0], target_size[1])
    
    print(f"Using {args.feature_extractor} feature extractor", flush=True)
    print(f"Target size: {target_size}", flush=True)
    print(f"State shape: {state_shape}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Run ID: {run_name}", flush=True)
    
    model_path = os.path.join(models_dir, f"DQN_{args.feature_extractor}{suffix}.pt")
    best_model_path = os.path.join(models_dir, f"DQN_{args.feature_extractor}{suffix}_best.pt")
    
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
        epsilon_decay=0.99995,
        model_dir=model_path,
        nb_hidden=hidden_size,
        feature_extractor=args.feature_extractor,
        finetune_features=args.finetune,
        target_size=target_size
    )

    agent = AgentUnified(
        **params.dict()
    )
    
    checkpoint_callback = create_checkpoint_callback(
        agent=agent,
        trainer=None,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        plots_dir=plots_dir,
        feature_extractor=args.feature_extractor,
        suffix=suffix,
        seed=seed,
        checkpoint_frequency=args.checkpoint_freq
    )
    
    trainer_args = {
        "n_episodes": args.episodes,
        "print_range": 100,
        "early_stop": args.early_stop,
        "max_timestep": 5000,
        "verbose": True,
        "checkpoint_callback": checkpoint_callback
    }

    print("Starting training...", flush=True)
    print(f"Checkpoints will be saved every {args.checkpoint_freq} episodes to {checkpoints_dir}", flush=True)
    trainer = Trainer(agent=agent, env=env, **trainer_args)
    
    try:
        trainer.run(
            logs_callback=agent.logs, 
            save_best_model=True, 
            output_path=best_model_path,
            run_dir=run_dir,
            agent_name=run_name
        )
        
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