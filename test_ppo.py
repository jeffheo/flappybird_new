import flappy_bird_gym
from src.ppo.state.agent import Agent
from src.ppo.state.config import AgentConfig
from src.utils import play_env
import numpy as np
import argparse

def main(model_dir, episodes):
    # Initialize environment
    env = flappy_bird_gym.make("FlappyBird-v0")
    
    # Initialize agent with same parameters
    params = AgentConfig(
        state_size=2,
        action_size=2,
        seed=0,
        model_dir=model_dir
    )

    # Create agent and load trained model
    agent = Agent(**params.dict())
    agent.load_model(model_dir)
    
    # Run episodes and collect scores
    scores = []
    for i in range(episodes):
        score = play_env(agent, env, fps=30, render=False)
        scores.append(score)
    
    # Print statistics
    print("\nEvaluation Results:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Standard Deviation: {np.std(scores):.2f}")
    print(f"Min Score: {min(scores):.2f}")
    print(f"Max Score: {max(scores):.2f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test PPO model')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    args = parser.parse_args()
    
    main(args.model_dir, args.episodes)