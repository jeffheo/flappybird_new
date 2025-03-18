import flappy_bird_gym
from src.dqn.agent import Agent
from src.dqn.config import AgentConfig
from src.utils import play_env
import numpy as np
import cv2

def preprocess_image(state):
    """Convert state to grayscale, resize, and normalize."""
    gray_state = np.mean(state, axis=2)
    
    resized_state = cv2.resize(gray_state, (84, 84), interpolation=cv2.INTER_AREA)
    
    normalized_state = resized_state / 255.0
    
    return normalized_state[np.newaxis, :, :]

def main():
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    env._renderer.make_display()  
    
    original_step = env.step
    original_reset = env.reset
    
    env.step = lambda action: tuple(map(
        lambda x: preprocess_image(x) if isinstance(x, np.ndarray) else x,
        original_step(action)
    ))
    env.reset = lambda: preprocess_image(original_reset())
    
    sample_state = env.reset()
    state_shape = sample_state.shape
    print(state_shape)
    print(sample_state)
    env.seed(100)
    params = AgentConfig(
        state_size=state_shape,
        action_size=2,
        seed=100,
        model_dir="models/DQN_image_final.pt",
    )

    agent = Agent(**params.dict())
    agent.load_model("models/DQN_image_best.pt")

    agent.epsilon_enabled = False

    scores = []
    for i in range(10):
        score = play_env(agent, env, fps=30, render=True)
        scores.append(score)
        print(f"Episode {i+1} Score: {score}")
    
    print("\nEvaluation Results:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Standard Deviation: {np.std(scores):.2f}")
    print(f"Min Score: {min(scores):.2f}")
    print(f"Max Score: {max(scores):.2f}")
    
    env.close()

if __name__ == "__main__":
    main() 