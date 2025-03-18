import time
import logging
import numpy as np
from gym import Env
# from src.base import Agent

_logger = logging.getLogger(__name__)

def play_env(agent, env: Env, fps: int = 30, render: bool = False):
    """
    Play an environment with an agent.

    Args:
    - agent: Agent to play the environment.
    - env: Environment to play.
    - fps: Frames per second.
    - render: Render the environment.
    """
    _logger.info("Playing environment...")
    score = 0
    
    # Initial reset
    state = env.reset()
    
    # Initialize display based on environment type
    if render:
        if 'rgb' in env.spec.id.lower():
            if env._renderer.display is None:
                env._renderer.make_display()
        else:
            env.render()
            time.sleep(0.5)  # Half second pause to ensure window appears
            env.close()
    
    while True:
        if render:
            env.render()
            time.sleep(1 / fps)
            
        action, log_prob, value = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state.copy()
        score += reward
        
        if done:
            break
    
    # Close display for state-based environment
    if 'rgb' not in env.spec.id.lower():
        env.close()
    
    _logger.info("Environment finished.")
    return score
