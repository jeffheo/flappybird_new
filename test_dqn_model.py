import numpy as np
import torch
import flappy_bird_gym
import argparse
import time
import os
from src.dqn.agent_unified import AgentUnified
from src.dqn.config_unified import UnifiedAgentConfig

def test_model(model_path, episodes=10, render=False, delay=0.01, feature_extractor='resnet', 
               frame_stack=False, frame_stack_size=2, target_size=(84, 84), 
               preprocess_method='enhanced', device=None):
    """
    Test a saved DQN model on the Flappy Bird environment.
    
    Args:
        model_path: Path to the saved model
        episodes: Number of episodes to run
        render: Whether to render the environment
        delay: Delay between frames (for visualization)
        feature_extractor: Feature extractor used in the model
        frame_stack: Whether frame stacking was used
        frame_stack_size: Number of frames stacked
        target_size: Image size used for preprocessing
        preprocess_method: Preprocessing method used
        device: Device to load the model to ('cuda', 'cpu')
    """
    # Disable pygame audio to avoid ALSA errors
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # Initialize environment
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    
    # Set random seed for reproducibility
    seed = 100
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Determine state shape based on preprocessing parameters
    if frame_stack:
        if feature_extractor in ['dinov2', 'clip', 'convnext']:
            # RGB input with frame stacking
            state_shape = (3 * frame_stack_size, target_size[0], target_size[1])
        else:
            # Grayscale input with frame stacking
            state_shape = (frame_stack_size, target_size[0], target_size[1])
    else:
        if feature_extractor in ['dinov2', 'clip', 'convnext']:
            # RGB input without frame stacking
            state_shape = (3, target_size[0], target_size[1])
        else:
            # Grayscale input without frame stacking
            state_shape = (1, target_size[0], target_size[1])
    
    # Initialize agent with the same parameters used during training
    params = UnifiedAgentConfig(
        state_size=state_shape,
        action_size=2,
        seed=seed,
        learning_rate=0.0001,  # Not used during testing
        batch_size=16,         # Not used during testing
        gamma=0.99,            # Not used during testing
        tau=0.001,             # Not used during testing
        epsilon_start=0.0,     # No exploration during testing
        epsilon_end=0.0,       # No exploration during testing
        epsilon_decay=1.0,     # No exploration during testing
        model_dir=model_path,
        feature_extractor=feature_extractor,
        finetune_features=False,  # Not relevant for testing
        use_frame_stack=frame_stack,
        frame_stack_size=frame_stack_size,
        target_size=target_size,
        preprocess_method=preprocess_method
    )
    
    # Create agent
    agent = AgentUnified(**params.dict())
    
    # Print model structure to identify the correct path to ResNet weights
    print("\n=== Model structure ===")
    print(agent.qnetwork_local)
    
    # Try to access and print some ResNet weights before loading
    print("\n=== ResNet weights BEFORE loading model ===")
    try:
        # Access the feature extractor (ResNet) weights
        if hasattr(agent.qnetwork_local, 'feature_extractor'):
            # Print first few weights from the first layer of ResNet
            for name, param in agent.qnetwork_local.feature_extractor.named_parameters():
                if 'weight' in name and param.requires_grad:
                    print(f"{name} - Shape: {param.shape}")
                    print(f"First few values: {param.data.flatten()[:5]}")
                    break
        else:
            # Alternative approach: print weights from any layer
            print("Feature extractor not found directly, printing weights from available layers:")
            for name, param in agent.qnetwork_local.named_parameters():
                if 'weight' in name:
                    print(f"{name} - Shape: {param.shape}")
                    print(f"First few values: {param.data.flatten()[:5]}")
                    break
    except Exception as e:
        print(f"Error accessing weights before loading: {e}")
    
    # Load the saved model
    try:
        # Use the device parameter from command line if provided, otherwise default to 'cpu' for testing
        agent.load_model(model_path, device=device if device else 'cpu')
        print(f"\nLoaded model from {model_path}")
        load_success = True
    except Exception as e:
        print(f"\nError loading model from {model_path}: {e}")
        print(f"Model file exists: {os.path.exists(model_path)}")
        print(f"Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'} bytes")
        load_success = False
    
    # Try to access and print some ResNet weights after loading
    print("\n=== ResNet weights AFTER loading model ===")
    try:
        # Access the feature extractor (ResNet) weights
        if hasattr(agent.qnetwork_local, 'feature_extractor'):
            # Print first few weights from the first layer of ResNet
            for name, param in agent.qnetwork_local.feature_extractor.named_parameters():
                if 'weight' in name and param.requires_grad:
                    print(f"{name} - Shape: {param.shape}")
                    print(f"First few values: {param.data.flatten()[:5]}")
                    break
        else:
            # Alternative approach: print weights from any layer
            print("Feature extractor not found directly, printing weights from available layers:")
            for name, param in agent.qnetwork_local.named_parameters():
                if 'weight' in name:
                    print(f"{name} - Shape: {param.shape}")
                    print(f"First few values: {param.data.flatten()[:5]}")
                    break
    except Exception as e:
        print(f"Error accessing weights after loading: {e}")
    
    # Test the agent
    scores = []
    
    for i in range(episodes):
        state = env.reset()
        agent.reset()  # Reset frame stack if used
        score = 0
        done = False
        
        while not done:
            # Get action from agent
            action = agent.act(state)  # Remove the train=False parameter
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Update state and score
            state = next_state
            score += reward
            
            # Render if requested
            if render:
                env.render()
                time.sleep(delay)  # Add delay for better visualization
        
        scores.append(score)
        # print(f"Episode {i+1}/{episodes}, Score: {score}")
    
    # Print statistics
    print("\nTesting Results:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Standard Deviation: {np.std(scores):.2f}")
    print(f"Min Score: {np.min(scores)}")
    print(f"Max Score: {np.max(scores)}")
    
    # Close environment
    env.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test a trained DQN model on Flappy Bird')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the saved model')
    parser.add_argument('--episodes', type=int, default=10, 
                        help='Number of episodes to test')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render the environment')
    parser.add_argument('--delay', type=float, default=0.01,
                        help='Delay between frames (for visualization)')
    parser.add_argument('--feature_extractor', type=str, default='resnet', 
                        choices=['spatial_cnn', 'resnet', 'efficientnet'],
                        help='Feature extractor used in the model')
    parser.add_argument('--frame_stack', action='store_true', default=False,
                        help='Whether frame stacking was used')
    parser.add_argument('--frame_stack_size', type=int, default=2,
                        help='Number of frames stacked')
    parser.add_argument('--target_size', type=str, default='224,224',
                        help='Target size for processed images (height,width)')
    parser.add_argument('--preprocess_method', type=str, default='enhanced',
                        choices=['basic', 'enhanced', 'context'],
                        help='Preprocessing method used')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to load the model to (\'cuda\', \'cpu\')')
    
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # Test the model
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