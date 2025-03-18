import argparse
import flappy_bird_gym
from src.dqn.agent import Agent
from src.dqn.config import AgentConfig
from src.utils_dqn import play_env_dqn
import numpy as np

def main(model_dir, episodes, render):
    # Initialize environment
    env = flappy_bird_gym.make("FlappyBird-v0")
    
    # Initialize agent with same parameters
    params = AgentConfig(
        state_size=2,
        action_size=2,
        seed=0,
        prioritized_memory=False,
        model_dir=model_dir
    )

    # Create agent and load trained model
    agent = Agent(**params.dict())
    agent.load_model(model_dir)
    
    # Disable exploration for evaluation
    agent.epsilon_enabled = False

    # Run 10 episodes and collect scores
    scores = []
    for i in range(episodes):
        score = play_env_dqn(agent, env, fps=30, render=render)
        scores.append(score)
    
    print("\nEvaluation Results:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Standard Deviation: {np.std(scores):.2f}")
    print(f"Min Score: {min(scores):.2f}")
    print(f"Max Score: {max(scores):.2f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DQN model')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--render', type=bool, default=False, help='Render the environment')
    args = parser.parse_args()
    
    main(args.model_dir, args.episodes, args.render) 