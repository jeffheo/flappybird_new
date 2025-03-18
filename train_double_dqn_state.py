import argparse
import numpy as np
import matplotlib.pyplot as plt
import flappy_bird_gym
import os
from datetime import datetime
from src.trainer import Trainer
from src.dqn.agent_double_dqn_state import AgentDoubleDQNState
from src.dqn.config_double_dqn_state import DoubleDQNStateConfig

def create_run_directory():
    """Create a unique directory for this training run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"DoubleDQN_state_run_{timestamp}"
    
    base_dir = "metrics"
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    return run_dir, models_dir, checkpoints_dir, run_name

def plot_scores(scores, window_size=100, save_path='training_progress.png'):
    plt.figure(figsize=(10, 6))
    
    plt.plot(scores, alpha=0.3, color='blue', label='Raw Scores')
    
    assert len(scores) >= window_size, f"Not enough episodes ({len(scores)}) to calculate {window_size}-episode moving average."
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(scores)), moving_avg, color='red', 
            label=f'{window_size}-episode Moving Average')
    
    
    plt.title('Double DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()

def save_training_results(agent, trainer, run_dir, models_dir, seed, final=False):
    """Save model, plot scores, and print statistics"""
    if final:
        final_model_path = os.path.join(models_dir, "DoubleDQN_state_final.pt")
        agent.save_model(final_model_path)
    
    plot_path = os.path.join(run_dir, "training_progress.png")
    plot_scores(trainer.scores, save_path=plot_path)
    
    print(f"\nTraining {'completed' if final else 'interrupted'}!")
    print(f"Best score: {max(trainer.scores)}")
    print(f"Average of last {min(100, len(trainer.scores))} episodes: {np.mean(trainer.scores[-min(100, len(trainer.scores)):]):.2f}")
    
    if trainer.metrics:
        stats = trainer.metrics.get_current_stats()
        print("\nMetrics Summary:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        try:
            trainer.metrics.save_metrics(additional_info={"final": final, "interrupted": not final, "seed": seed}, generate_plots=True)
            print(f"Metrics saved successfully to {run_dir}")
        except Exception as e:
            print(f"Error saving metrics: {e}")

def create_checkpoint_callback(agent, run_dir, checkpoints_dir, seed, checkpoint_frequency=10):
    """
    Create a callback function for periodic checkpointing.
    
    Args:
        agent: The agent to save
        run_dir: Directory to save metrics
        checkpoints_dir: Directory to save model checkpoints
        seed: Random seed used for training
        checkpoint_frequency: How often to save checkpoints (in episodes)
        
    Returns:
        checkpoint_callback: Function to be called after each episode
    """
    def checkpoint_callback(trainer, episode):
        if episode % checkpoint_frequency != 0:
            return
            
        try:
            checkpoint_path = os.path.join(checkpoints_dir, f"DoubleDQN_state_checkpoint_ep{episode}.pt")
            agent.save_model(checkpoint_path)
            
            if trainer.metrics:
                trainer.metrics.save_metrics(additional_info={
                    "checkpoint": True,
                    "episode": episode,
                    "seed": seed
                }, generate_plots=False)
            
            print(f"\nCheckpoint saved at episode {episode}")
                
        except Exception as e:
            print(f"Error during checkpointing at episode {episode}: {e}")
    
    return checkpoint_callback

def main():
    parser = argparse.ArgumentParser(description='Train a Double DQN model on Flappy Bird')
    parser.add_argument('--seed', type=int, default=1, help='Seed for the environment')
    args = parser.parse_args()
    
    run_dir, models_dir, checkpoints_dir, run_name = create_run_directory()
    print(f"Training results will be saved to: {run_dir}")
    
    env = flappy_bird_gym.make("FlappyBird-v0")
    
    seed = args.seed
    env.seed(seed)
    params = DoubleDQNStateConfig(
        state_size=2,
        action_size=2,
        seed=seed,
        prioritized_memory=False,
        model_dir=os.path.join(models_dir, "DoubleDQN_state.pt"),
        learning_rate=0.0005,
        batch_size=64,
        gamma=0.99,
        tau=0.001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99995
    )

    agent = AgentDoubleDQNState(**params.dict())
    
    checkpoint_frequency = 10000
    
    checkpoint_callback = create_checkpoint_callback(
        agent=agent,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        seed=seed,
        checkpoint_frequency=checkpoint_frequency
    )
    
    trainer_args = {
        "n_episodes": 100000,
        "print_range": 1000,
        "early_stop": 5000,
        "max_timestep": 5000,
        "verbose": True,
        "checkpoint_callback": checkpoint_callback
    }

    print("Starting training...")
    print(f"Checkpoints will be saved every {checkpoint_frequency} episodes to {checkpoints_dir}")
    
    trainer = Trainer(agent=agent, env=env, **trainer_args)
    
    try:
        trainer.run(
            logs_callback=agent.logs, 
            save_best_model=True, 
            output_path=os.path.join(models_dir, "DoubleDQN_state_best.pt"),
            run_dir=run_dir,
            agent_name=run_name
        )
        
        save_training_results(agent, trainer, run_dir, models_dir, seed, final=True)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        save_training_results(agent, trainer, run_dir, models_dir, seed, final=False)
    finally:
        env.close()

if __name__ == "__main__":
    main() 