import time
from gym import Env

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
        
        act_result = agent.act(state)
        
        if hasattr(act_result, '__iter__') and not isinstance(act_result, (str, bytes, bytearray)): # to account for architecture diffs
            action = act_result[0]
        else:
            action = act_result
            
        next_state, reward, done, _ = env.step(action)
        state = next_state.copy()
        score += reward
        
        if done:
            break
    if 'rgb' not in env.spec.id.lower():
        env.close()
    
    return score 