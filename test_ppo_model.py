import numpy as np
import torch
import flappy_bird_gym
import argparse
import time
import os
from src.ppo.rgb.agent import Agent
from src.metrics import MetricsTracker

def test_model(model_path, episodes=10, render=False, delay=0.01, feature_extractor='resnet', 
               frame_stack=True, frame_stack_size=4, target_size=(84, 84), 
               preprocess_method='enhanced', device=None):
    
    # Disable pygame audio to avoid ALSA errors
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    
    seed = 100
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Determine state shape based on preprocessing parameters
    if frame_stack:
        state_shape = (frame_stack_size, target_size[0], target_size[1])
    else:
        state_shape = (1, target_size[0], target_size[1])
    
    # Initialize agent with the same parameters used during training
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
        max_grad_norm=0.5,     # Not used during testing
        update_interval=2048,  # Not used during testing
        model_dir=model_path,
        feature_extractor=feature_extractor,
        finetune_features=True,  # Not relevant for testing
        use_frame_stack=frame_stack,
        frame_stack_size=frame_stack_size,
        target_size=target_size,
        preprocess_method=preprocess_method
    )
    
    metrics_tracker = MetricsTracker(agent_name="PPO_Test")
    agent.set_metrics_tracker(metrics_tracker)
    
    print("\n=== Model structure ===")
    print(agent.actor_critic)
    
    print("\n=== Parameter counts ===")
    total_params = sum(p.numel() for p in agent.actor_critic.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Count parameters by component
    if hasattr(agent.actor_critic, 'features'):
        feature_params = sum(p.numel() for p in agent.actor_critic.features.parameters())
        print(f"Feature extractor parameters: {feature_params:,}")
    
    if hasattr(agent.actor_critic, 'actor'):
        actor_params = sum(p.numel() for p in agent.actor_critic.actor.parameters())
        print(f"Actor network parameters: {actor_params:,}")
    
    if hasattr(agent.actor_critic, 'critic'):
        critic_params = sum(p.numel() for p in agent.actor_critic.critic.parameters())
        print(f"Critic network parameters: {critic_params:,}")
    
    # Count trainable vs non-trainable parameters
    trainable_params = sum(p.numel() for p in agent.actor_critic.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    print("\n=== Feature extractor weights BEFORE loading model ===")
    try:
        # Access the feature extractor weights
        if hasattr(agent.actor_critic, 'features'):
            # Print first few weights from the first layer of feature extractor
            for name, param in agent.actor_critic.features.named_parameters():
                if 'weight' in name and param.requires_grad:
                    print(f"{name} - Shape: {param.shape}")
                    print(f"First few values: {param.data.flatten()[:5]}")
                    break
        else:
            # Alternative approach: print weights from any layer
            print("Feature extractor not found directly, printing weights from available layers:")
            for name, param in agent.actor_critic.named_parameters():
                if 'weight' in name:
                    print(f"{name} - Shape: {param.shape}")
                    print(f"First few values: {param.data.flatten()[:5]}")
                    break
    except Exception as e:
        print(f"Error accessing weights before loading: {e}")
    
    # Load the saved model
    try:
        device_to_use = device if device else 'cpu'
        if device:
            agent.device = torch.device(device)
            agent.actor_critic = agent.actor_critic.to(agent.device)
        agent.load_model(model_path)
        print(f"\nLoaded model from {model_path}")
        load_success = True
    except Exception as e:
        print(f"\nError loading model from {model_path}: {e}")
        print(f"Model file exists: {os.path.exists(model_path)}")
        print(f"Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'} bytes")
        load_success = False
    
    print("\n=== Feature extractor weights AFTER loading model ===")
    try:
        if hasattr(agent.actor_critic, 'features'):
            # Print first few weights from the first layer of feature extractor
            for name, param in agent.actor_critic.features.named_parameters():
                if 'weight' in name and param.requires_grad:
                    print(f"{name} - Shape: {param.shape}")
                    print(f"First few values: {param.data.flatten()[:5]}")
                    break
        else:
            # Alternative approach: print weights from any layer
            print("Feature extractor not found directly, printing weights from available layers:")
            for name, param in agent.actor_critic.named_parameters():
                if 'weight' in name:
                    print(f"{name} - Shape: {param.shape}")
                    print(f"First few values: {param.data.flatten()[:5]}")
                    break
    except Exception as e:
        print(f"Error accessing weights after loading: {e}")
    
    print("\n=== Optimizer parameter groups ===")
    try:
        if hasattr(agent, 'optimizer'):
            print(f"Number of parameter groups: {len(agent.optimizer.param_groups)}")
            for i, group in enumerate(agent.optimizer.param_groups):
                print(f"Group {i}: {len(group['params'])} parameters, lr={group['lr']}")
                
                param_count = sum(p.numel() for p in group['params'])
                print(f"  Total parameters in group: {param_count:,}")
                
                print("  Sample parameters from:")
                param_to_name = {}
                for name, param in agent.actor_critic.named_parameters():
                    param_to_name[param] = name
                
                # Sample up to 3 parameters from this group to show which modules they're from
                samples = 0
                for param in group['params']:
                    if param in param_to_name and samples < 3:
                        print(f"    - {param_to_name[param]}")
                        samples += 1
        else:
            print("No optimizer found in agent")
    except Exception as e:
        print(f"Error examining optimizer: {e}")
    
    scores = []
    
    if render:
        if 'rgb' in env.spec.id.lower():
            if env._renderer.display is None:
                env._renderer.make_display()

    for i in range(episodes):
        state = env.reset()
        agent.preprocessor.reset()  # Reset frame stack if used
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
        # print(f"Episode {i+1}/{episodes}, Score: {score}")
    
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
    parser.add_argument('--frame_stack', action='store_true', default=False,
                        help='Whether frame stacking was used')
    parser.add_argument('--frame_stack_size', type=int, default=4,
                        help='Number of frames stacked')
    parser.add_argument('--target_size', type=str, default='224,224',
                        help='Target size for processed images (height,width)')
    parser.add_argument('--preprocess_method', type=str, default='enhanced',
                        choices=['basic', 'enhanced'],
                        help='Preprocessing method used')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to load the model to (\'cuda\', \'cpu\')')
    
    args = parser.parse_args()
    
    target_size = tuple(map(int, args.target_size.split(',')))
    
    test_model(
        model_path=args.model_path,
        episodes=args.episodes,
        render=args.render,
        delay=args.delay,
        feature_extractor=args.feature_extractor,
        frame_stack=args.frame_stack,
        frame_stack_size=args.frame_stack_size,
        target_size=target_size,
        preprocess_method=args.preprocess_method,
        device=args.device
    )

if __name__ == "__main__":
    main() 