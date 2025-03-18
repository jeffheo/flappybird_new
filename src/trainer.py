import numpy as np
from collections import deque
import gc
import torch
from src.metrics import MetricsTracker

class Trainer:
    def __init__(self, agent, env, n_episodes=10000, print_range=100, early_stop=None, max_timestep=None, verbose=True, checkpoint_callback=None):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.print_range = print_range
        self.early_stop = early_stop
        self.max_timestep = max_timestep
        self.verbose = verbose
        self.checkpoint_callback = checkpoint_callback
        
        self.scores = []
        self.scores_window = deque(maxlen=print_range)
        self.best_score = -np.inf
        self.last_episode = 0
        
        self.use_cuda = torch.cuda.is_available()
        
        self.metrics = None

    def run(self, logs_callback=None, save_best_model=False, output_path=None, agent_name="DQN", run_dir=None):
        
        self.metrics = MetricsTracker(agent_name=agent_name, window_size=self.print_range, save_dir=run_dir)
        
        if hasattr(self.agent, 'set_metrics_tracker'):
            self.agent.set_metrics_tracker(self.metrics)
        
        for i_episode in range(1, self.n_episodes + 1):
            try:
                state = self.env.reset()
                if hasattr(self.agent, 'reset'):
                    self.agent.reset()
                
                score = 0
                timestep = 0
                
                self.metrics.start_episode()
                
                while True:
                    try:
                        action = self.agent.act(state)
                        next_state, reward, done, _ = self.env.step(action)
                        self.agent.step(state, action, reward, next_state, done)
                        
                        # Track exploration (epsilon for DQN)
                        if hasattr(self.agent, 'epsilon'):
                            self.metrics.log_exploration(self.agent.epsilon)
                        
                        state = next_state
                        score += reward
                        timestep += 1
                        
                        if done or (self.max_timestep and timestep >= self.max_timestep):
                            break
                    except Exception as e:
                        print(f"Error during episode step: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        break
                
                self.metrics.end_episode(score, timestep)
                
                self.scores_window.append(score)
                self.scores.append(score)
                
                if score > self.best_score:
                    self.best_score = score
                    if save_best_model and output_path:
                        try:
                            self.agent.save_model(output_path)
                            if self.verbose:
                                print(f"\nNew best score: {score}! Model saved to {output_path}", flush=True)
                        except Exception as e:
                            print(f"Error saving best model: {e}", flush=True)
                
                if self.verbose and i_episode % self.print_range == 0:
                    print(f'Episode {i_episode}\tAverage Score: {np.mean(self.scores_window):.2f}', flush=True)
                    
                    stats = self.metrics.get_current_stats()
                    if 'grad_norm_mean' in stats:
                        print(f'Gradient Norm: {stats["grad_norm_mean"]:.4f}', flush=True)
                    if 'total_loss_mean' in stats:
                        print(f'Loss: {stats["total_loss_mean"]:.4f}', flush=True)
                    if 'exploration' in stats:
                        print(f'Exploration Rate: {stats["exploration"]:.4f}', flush=True)
                    
                    if logs_callback:
                        logs_callback()
                
                if self.checkpoint_callback:
                    try:
                        self.checkpoint_callback(self, i_episode)
                    except Exception as e:
                        print(f"Error in checkpoint callback: {e}", flush=True)

                if self.early_stop and np.mean(self.scores_window) >= self.early_stop:
                    if self.verbose:
                        print(f'\nEnvironment solved in {i_episode} episodes!', flush=True)
                    break
                
                if i_episode % 10 == 0:
                    gc.collect()
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                
                self.last_episode = i_episode
            except Exception as e:
                print(f"Error during episode {i_episode}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
        
        try:
            self.metrics.save_metrics(additional_info={"final": True})
        except Exception as e:
            print(f"Error saving final metrics: {e}", flush=True) 