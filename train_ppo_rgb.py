import numpy as np
import matplotlib.pyplot as plt
import flappy_bird_gym
import torch
import argparse
import os
from datetime import datetime
from src.ppo_trainer import PPOTrainer
from src.ppo.rgb.agent import Agent
from src.ppo.rgb.config import AgentConfig

def create_run_directory():
    """Create a unique directory for this training run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_RGB_run_{timestamp}"
    
    base_dir = "metrics"
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)    
    return run_dir, models_dir, checkpoints_dir, run_name

def plot_scores(scores, window_size=100, save_path='ppo_rgb_training_progress.png'):
    """Plot and save the learning curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.3, color='blue', label='Raw Scores')
    
    assert len(scores) >= window_size, f"Not enough episodes ({len(scores)}) to calculate {window_size}-episode moving average."
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(scores)), moving_avg, color='red', 
            label=f'{window_size}-episode Moving Average')
    plt.title('PPO Training Progress (RGB Observations)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_training_results(agent, trainer, run_dir, models_dir, feature_extractor, model_size, seed, final=False):
    """Save model, plot scores, and print statistics"""
    if final:
        final_model_path = os.path.join(models_dir, f"PPO_RGB_{feature_extractor}_{model_size}_final.pt")
        agent.save_model(final_model_path)
    
    plot_path = os.path.join(run_dir, "ppo_rgb_training_progress.png")
    plot_scores(trainer.scores, save_path=plot_path)
    
    print(f"\nTraining {'completed' if final else 'interrupted'}!")
    print(f"Best score: {max(trainer.scores)}")
    print(f"Average of last {min(100, len(trainer.scores))} episodes: {np.mean(trainer.scores[-min(100, len(trainer.scores)):]):.2f}")
    print(f"Total timesteps: {trainer.total_timesteps}")
    
    if trainer.metrics:
        stats = trainer.metrics.get_current_stats()
        print("\nMetrics Summary:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        try:
            trainer.metrics.save_metrics(additional_info={
                "final": final, 
                "interrupted": not final, 
                "seed": seed,
                "feature_extractor": feature_extractor,
                "model_size": model_size
            }, generate_plots=True)
            print(f"Metrics saved successfully to {run_dir}")
        except Exception as e:
            print(f"Error saving metrics: {e}")

def create_checkpoint_callback(agent, run_dir, checkpoints_dir, feature_extractor, model_size, seed, checkpoint_frequency=10):
    """
    Create a callback function for periodic checkpointing.
    
    Args:
        agent: The agent to save
        run_dir: Directory to save metrics
        checkpoints_dir: Directory to save model checkpoints
        feature_extractor: Type of feature extractor used
        model_size: Size of the foundation model
        seed: Random seed used for training
        checkpoint_frequency: How often to save checkpoints (in episodes)
        
    Returns:
        checkpoint_callback: Function to be called after each episode
    """
    def checkpoint_callback(trainer, episode):
        if episode % checkpoint_frequency != 0:
            return
            
        try:
            checkpoint_path = os.path.join(checkpoints_dir, f"PPO_RGB_{feature_extractor}_{model_size}_checkpoint_ep{episode}.pt")
            agent.save_model(checkpoint_path)
            
            if trainer.metrics:
                trainer.metrics.save_metrics(additional_info={
                    "checkpoint": True,
                    "episode": episode,
                    "seed": seed,
                    "feature_extractor": feature_extractor,
                    "model_size": model_size
                }, generate_plots=False)
            
            print(f"\nCheckpoint saved at episode {episode}")
                
        except Exception as e:
            print(f"Error during checkpointing at episode {episode}: {e}")
    
    return checkpoint_callback

def main():
    parser = argparse.ArgumentParser(description='Train PPO agent on Flappy Bird with RGB observations')
    parser.add_argument('--feature_extractor', type=str, default='resnet',
                        choices=['spatial_cnn', 'resnet'],
                        help='Feature extraction method')
    parser.add_argument('--finetune_features', action='store_true',
                        help='Whether to fine-tune the feature extractor')
    parser.add_argument('--target_height', type=int, default=224,
                        help='Target image height')
    parser.add_argument('--target_width', type=int, default=224,
                        help='Target image width')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--update_interval', type=int, default=512,
                        help='Steps before PPO update')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--checkpoint_frequency', type=int, default=10000,
                        help='Episodes between checkpoints')
    
    args = parser.parse_args()
    
    run_dir, models_dir, checkpoints_dir, run_name = create_run_directory()
    print(f"Training results will be saved to: {run_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    
    seed = args.seed
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    channels = 1
    
    params = AgentConfig(
        state_size=(channels, args.target_height, args.target_width),
        action_size=2,
        seed=seed,
        nb_hidden=(64,64),
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=4,
        value_coef=0.5,
        entropy_coef=0.02,
        max_grad_norm=0.5,
        update_interval=1024,
        model_dir=os.path.join(models_dir, "PPO_RGB.pt"),
        feature_extractor=args.feature_extractor,
        finetune_features=args.finetune_features,
        target_size=(224,224),
    )

    agent = Agent(**params.dict())
    
    checkpoint_callback = create_checkpoint_callback(
        agent=agent,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        feature_extractor=args.feature_extractor,
        seed=seed,
        checkpoint_frequency=args.checkpoint_frequency
    )
    
    trainer_args = {
        "n_episodes":  100000,
        "print_range": 100,
        "early_stop": 5000,
        "max_timestep": 5000,
        "verbose": True,
        "checkpoint_callback": checkpoint_callback
    }

    print("Starting PPO RGB training...")
    print(f"Checkpoints will be saved every {args.checkpoint_frequency} episodes to {checkpoints_dir}")
    
    trainer = PPOTrainer(agent=agent, env=env, **trainer_args)
    
    try:
        trainer.run(
            logs_callback=agent.logs, 
            save_best_model=True, 
            output_path=os.path.join(models_dir, f"PPO_RGB_{args.feature_extractor}_best.pt"),
            run_dir=run_dir,
            agent_name=run_name
        )
        
        save_training_results(
            agent, 
            trainer, 
            run_dir, 
            models_dir, 
            args.feature_extractor, 
            seed, 
            final=True
        )
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        save_training_results(
            agent, 
            trainer, 
            run_dir, 
            models_dir, 
            args.feature_extractor, 
            seed, 
            final=False
        )
    finally:
        env.close()

if __name__ == "__main__":
    main()