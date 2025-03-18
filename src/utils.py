# Inspired by utility code from https://github.com/DougTrajano/drl-flappy-bird/blob/main/src/utils.py
import time
from gym import Env

def play_env(agent, env: Env, fps: int = 30, render: bool = False):
    """
    Play an environment with an agent.

    Args:
    - agent: Agent to play the environment.
    - env: Environment to play.
    - fps: Frames per second.
    - render: Render the environment.
    """
    score = 0
    
    state = env.reset()
    
    if render:
        if 'rgb' in env.spec.id.lower():
            if env._renderer.display is None:
                env._renderer.make_display()
        else:
            env.render()
            time.sleep(0.5)
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
    
    if 'rgb' not in env.spec.id.lower():
        env.close()
    
    return score
