import argparse
import flappy_bird_gym
from src.dqn.agent_double_dqn_state import AgentDoubleDQNState
from src.dqn.config_double_dqn_state import DoubleDQNStateConfig
from src.utils_dqn import play_env_dqn
import numpy as np

def main(model_dir, episodes, render=False, seed=100):
    env = flappy_bird_gym.make("FlappyBird-v0")
    env.seed(seed)
    params = DoubleDQNStateConfig(
        state_size=2,
        action_size=2,
        seed=seed,
        model_dir=model_dir
    )

    agent = AgentDoubleDQNState(**params.dict())
    agent.load_model(model_dir)
    
    agent.epsilon_enabled = False

    scores = []
    for i in range(episodes):
        print(f"Episode {i+1}/{episodes}")
        score = play_env_dqn(agent, env, fps=30, render=render)
        scores.append(score)
    
    print("\nEvaluation Results:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Standard Deviation: {np.std(scores):.2f}")
    print(f"Min Score: {min(scores):.2f}")
    print(f"Max Score: {max(scores):.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Double DQN model with state-based observations')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--seed', type=int, default=100, help='Seed for the environment')
    args = parser.parse_args()
    
    main(args.model_dir, args.episodes, args.render, args.seed) 