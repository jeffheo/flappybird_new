import numpy as np
from collections import deque
from src.metrics import MetricsTracker

class PPOTrainer:
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
        
        self.total_timesteps = 0
        
        self.metrics = None

    def run(self, logs_callback=None, save_best_model=False, output_path=None, agent_name="PPO", run_dir=None):
        self.metrics = MetricsTracker(agent_name=agent_name, window_size=self.print_range, save_dir=run_dir)
        
        if hasattr(self.agent, 'set_metrics_tracker'):
            self.agent.set_metrics_tracker(self.metrics)
        
        for i_episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            score = 0
            done = False
            timestep = 0
            
            self.metrics.start_episode()
            
            while not done:
                action, log_prob, value = self.agent.act(state)
                
                next_state, reward, done, _ = self.env.step(action)
                
                self.agent.step(state, action, reward, next_state, done)
                
                state = next_state
                score += reward
                self.total_timesteps += 1
                timestep += 1
                
                # Break if max timestep is reached
                if self.max_timestep and timestep >= self.max_timestep:
                    break
            
            self.metrics.end_episode(score, timestep)
            
            self.scores_window.append(score)
            self.scores.append(score)
            
            if score > self.best_score:
                self.best_score = score
                if save_best_model and output_path:
                    self.agent.save_model(output_path)
                    if self.verbose:
                        print(f"\nNew best score: {score}! Model saved to {output_path}")
            
            if self.verbose and i_episode % self.print_range == 0:
                print(f'Episode {i_episode}\tAverage Score: {np.mean(self.scores_window):.2f}\tTotal Timesteps: {self.total_timesteps}')
                
                stats = self.metrics.get_current_stats()
                if 'exploration' in stats:
                    print(f'Exploration Rate: {stats["exploration"]:.4f}')
                
                if logs_callback:
                    logs_callback()
            
            if self.checkpoint_callback:
                try:
                    self.checkpoint_callback(self, i_episode)
                except Exception as e:
                    print(f"Error in checkpoint callback: {e}")
            
            if self.early_stop and np.mean(self.scores_window) >= self.early_stop:
                if self.verbose:
                    print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(self.scores_window):.2f}')
                break
            
            self.last_episode = i_episode
        
        try:
            self.metrics.save_metrics(additional_info={"final": True})
        except Exception as e:
            print(f"Error saving final metrics: {e}")