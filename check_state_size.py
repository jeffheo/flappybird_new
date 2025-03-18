import numpy as np
import flappy_bird_gym
import os
import matplotlib.pyplot as plt
import argparse

def check_state_size(render=True, save_image=False):
    """
    Check the pixel size of the state (image) returned by the Flappy Bird environment.
    
    Args:
        render: Whether to render the environment
        save_image: Whether to save the first frame as an image
    """
    # Disable pygame audio to avoid ALSA errors
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # Initialize environment
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    
    # Reset environment to get initial state
    state = env.reset()
    
    # Print state information
    print("\nState Information:")
    print(f"Type: {type(state)}")
    print(f"Shape: {state.shape}")
    print(f"Data type: {state.dtype}")
    print(f"Min value: {np.min(state)}")
    print(f"Max value: {np.max(state)}")
    
    memory_bytes = state.nbytes
    memory_kb = memory_bytes / 1024
    memory_mb = memory_kb / 1024
    
    print(f"\nMemory usage:")
    print(f"Bytes: {memory_bytes}")
    print(f"Kilobytes: {memory_kb:.2f} KB")
    print(f"Megabytes: {memory_mb:.4f} MB")
    
    if len(state.shape) == 3 and state.shape[2] == 3:
        print("\nRGB Channel Information:")
        print(f"Red channel shape: {state[:, :, 0].shape}")
        print(f"Green channel shape: {state[:, :, 1].shape}")
        print(f"Blue channel shape: {state[:, :, 2].shape}")
        
        print(f"Average Red value: {np.mean(state[:, :, 0]):.2f}")
        print(f"Average Green value: {np.mean(state[:, :, 1]):.2f}")
        print(f"Average Blue value: {np.mean(state[:, :, 2]):.2f}")
    
    if save_image:
        plt.figure(figsize=(10, 8))
        plt.imshow(state)
        plt.title(f"Flappy Bird State - Shape: {state.shape}")
        plt.axis('on')  
        plt.colorbar(label='Pixel Value')
        plt.savefig('flappy_bird_state.png')
        print("\nSaved state image to 'flappy_bird_state.png'")
    
    if render:
        env.render()
        input("\nPress Enter to continue...")
    
    action = np.random.randint(0, 2)  # 0 or 1
    next_state, _, _, _ = env.step(action)
    
    print("\nNext State Information:")
    print(f"Shape: {next_state.shape}")
    print(f"Same shape as initial state: {next_state.shape == state.shape}")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='Check the pixel size of the state in Flappy Bird')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render the environment')
    parser.add_argument('--save_image', action='store_true', default=False,
                        help='Whether to save the first frame as an image')
    
    args = parser.parse_args()
    
    check_state_size(render=args.render, save_image=args.save_image)

if __name__ == "__main__":
    main() 