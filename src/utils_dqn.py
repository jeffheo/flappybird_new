import time
import logging
from gym import Env
import inspect

_logger = logging.getLogger(__name__)

def play_env_dqn(agent, env: Env, fps: int = 30, render: bool = False):
    """
    Play an environment with a DQN-based agent.
    This function handles agents whose act() method returns either:
    - Just an action (like DQN, Double DQN)
    - A tuple of (action, log_prob, value) (like PPO)

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
        
        # Call agent.act() and handle different return types
        act_result = agent.act(state)
        
        # Check if act_result is a tuple-like object or a single value
        if hasattr(act_result, '__iter__') and not isinstance(act_result, (str, bytes, bytearray)):
            # Unpack tuple (action, log_prob, value)
            action = act_result[0]
        else:
            # Single value (just the action)
            action = act_result
            
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