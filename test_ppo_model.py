import numpy as np
import torch
import flappy_bird_gym
import argparse
import time
import os
from src.ppo.rgb.agent import Agent
from src.metrics import MetricsTracker

def test_model(model_path, episodes=10, render=False, delay=0.01, feature_extractor='resnet', target_size=(224, 224), 
               device=None, seed=100):
    
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    state_shape = (1, target_size[0], target_size[1])
    
    agent = Agent(
        state_size=state_shape,
        action_size=2,
        seed=seed,
        learning_rate=0.0003,  # Not used during testing
        gamma=0.99,            # Not used during testing
        gae_lambda=0.95,       # Not used during testing
        policy_clip=0.2,       # Not used during testing
        batch_size=64,         # Not used during testing
        n_epochs=10,           # Not used during testing
        value_coef=0.5,        # Not used during testing
        entropy_coef=0.01,     # Not used during testing
        model_dir=model_path,
        feature_extractor=feature_extractor,
        finetune_features=True,  # Not relevant for testing
        target_size=target_size,
    )
    
    metrics_tracker = MetricsTracker(agent_name="PPO_Test")
    agent.set_metrics_tracker(metrics_tracker)
    
    # Load the saved model
    try:
        if device:
            agent.device = torch.device(device)
            agent.actor_critic = agent.actor_critic.to(agent.device)
        agent.load_model(model_path)
        print(f"\nLoaded model from {model_path}")
    except Exception as e:
        print(f"\nError loading model from {model_path}: {e}")
        print(f"Model file exists: {os.path.exists(model_path)}")
        print(f"Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'} bytes")
    
    scores = []
    
    if render:
        if 'rgb' in env.spec.id.lower():
            if env._renderer.display is None:
                env._renderer.make_display()

    for i in range(episodes):
        state = env.reset()
        score = 0
        done = False
        
        while not done:
            action, _, _ = agent.act(state)
            
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            score += reward
            
            if render:
                print("Rendering...")
                env.render()
                time.sleep(delay)  
        
        scores.append(score)
    
    print("\nTesting Results:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Standard Deviation: {np.std(scores):.2f}")
    print(f"Min Score: {np.min(scores)}")
    print(f"Max Score: {np.max(scores)}")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='Test a trained PPO model on Flappy Bird')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the saved model')
    parser.add_argument('--episodes', type=int, default=10, 
                        help='Number of episodes to test')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render the environment')
    parser.add_argument('--delay', type=float, default=0.01,
                        help='Delay between frames (for visualization)')
    parser.add_argument('--feature_extractor', type=str, default='resnet', 
                        choices=['spatial_cnn', 'resnet'],
                        help='Feature extractor used in the model')
    parser.add_argument('--target_size', type=str, default='224,224',
                        help='Target size for processed images (height,width)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to load the model to (\'cuda\', \'cpu\')')
    parser.add_argument('--seed', type=int, default=100,
                        help='Seed for the environment')
    args = parser.parse_args()
    
    target_size = tuple(map(int, args.target_size.split(',')))
    
    test_model(
        model_path=args.model_path,
        episodes=args.episodes,
        render=args.render,
        delay=args.delay,
        feature_extractor=args.feature_extractor,
        target_size=target_size,
        device=args.device,
        seed=args.seed
    )

if __name__ == "__main__":
    main() 