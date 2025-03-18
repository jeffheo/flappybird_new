import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=(64, 64)):
        super(ActorCritic, self).__init__()
        
        # Actor network (policy)
        self.actor_fc1 = nn.Linear(state_size, hidden_size[0])
        self.actor_fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.actor_fc3 = nn.Linear(hidden_size[1], action_size)
        
        # Critic network (value function)
        self.critic_fc1 = nn.Linear(state_size, hidden_size[0])
        self.critic_fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.critic_fc3 = nn.Linear(hidden_size[1], 1)
        
    def actor(self, x):
        x = F.relu(self.actor_fc1(x))
        x = F.relu(self.actor_fc2(x))
        return F.softmax(self.actor_fc3(x), dim=-1)
    
    def critic(self, x):
        x = F.relu(self.critic_fc1(x))
        x = F.relu(self.critic_fc2(x))
        return self.critic_fc3(x)
    
    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states), np.array(self.actions), \
               np.array(self.probs), np.array(self.vals), \
               np.array(self.rewards), np.array(self.dones), \
               batches

class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        seed=0,
        nb_hidden=(64, 64),
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_interval=512,
        model_dir="../models/PPO.pt"
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_interval = update_interval
        
        self.actor_critic = ActorCritic(state_size, action_size, hidden_size=nb_hidden).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        self.memory = PPOMemory(batch_size)
        
        self.step_counter = 0
        
        self.model_dir = model_dir
        
        self.last_policy_loss = 0
        self.last_value_loss = 0
        self.last_entropy = 0
        self.last_total_loss = 0
        self.last_clipped_updates = 0
        self.last_update_ratio = 0
        self.metrics_tracker = None
    
    def act(self, state, eval_mode=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        else:
            state = state.float().to(self.device)
        
        self.actor_critic.eval()
        with torch.no_grad():
            probs, value = self.actor_critic(state)
        
        # Only use for training, not for evaluation
        if not eval_mode:
            dist = Categorical(probs)
            action = dist.sample().item()
            
            return action, 0, 0
        else:
            return torch.argmax(probs).item(), 0, 0
    
    def evaluate(self, states, actions):
        """Evaluate actions given states."""
        probs, values = self.actor_critic(states)
        
        dist = Categorical(probs)
        
        action_log_probs = dist.log_prob(actions)
        
        dist_entropy = dist.entropy()
        
        return action_log_probs, values, dist_entropy
    
    def step(self, state, action, reward, next_state, done):
        """Store experience in memory and learn if enough steps have been taken."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        
        self.actor_critic.eval()
        with torch.no_grad():
            probs, value = self.actor_critic(state)
        
        action_prob = probs[action].item() if probs.dim() == 1 else probs[0, action].item()
        self.memory.store(state.cpu().numpy(), action, action_prob, value.item(), reward, done)
        
        self.step_counter += 1
        
        if self.step_counter >= self.update_interval:
            self.learn()
            self.step_counter = 0
    
    def learn(self):
        states, actions, old_probs, vals, rewards, dones, batches = self.memory.generate_batches()
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(self.device)
        
        advantages = self._compute_gae(states, vals, rewards, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = advantages + torch.tensor(vals, dtype=torch.float32).to(self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_loss = 0
        total_clipped = 0
        total_update_ratio = 0
        
        for _ in range(self.n_epochs):
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_probs = old_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = returns[batch]
                
                new_log_probs, critic_values, entropy = self.evaluate(batch_states, batch_actions)
                
                prob_ratio = torch.exp(new_log_probs - torch.log(batch_old_probs))
                
                # Compute surrogate losses
                surrogate1 = prob_ratio * batch_advantages
                surrogate2 = torch.clamp(prob_ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                
                # Count clipped updates for metrics
                clipped = (prob_ratio < 1.0 - self.policy_clip).sum().item() + (prob_ratio > 1.0 + self.policy_clip).sum().item()
                total_clipped += clipped
                
                # Calculate average update ratio for metrics
                update_ratio = torch.abs(prob_ratio - 1.0).mean().item()
                total_update_ratio += update_ratio
                
                # Compute actor (policy) loss
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Compute critic (value) loss - fix shape mismatch
                # Ensure both tensors have the same shape by squeezing critic_values
                critic_loss = nn.MSELoss()(critic_values.squeeze(), batch_returns)
                
                entropy_loss = entropy.mean()
                
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss
                
                total_policy_loss += actor_loss.item()
                total_value_loss += critic_loss.item()
                total_entropy += entropy_loss.item()
                total_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                
                grad_norm = 0
                for param in self.actor_critic.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                
                if hasattr(self, 'metrics_tracker') and self.metrics_tracker is not None:
                    self.metrics_tracker.log_loss(
                        total=loss.item(),
                        policy=actor_loss.item(),
                        value=critic_loss.item(),
                        entropy=entropy_loss.item()
                    )
                    self.metrics_tracker.log_param_stats(
                        self.actor_critic,
                        grad_norm=grad_norm,
                        update_ratio=update_ratio,
                        clipped_updates=clipped
                    )
                    self.metrics_tracker.log_exploration(entropy_loss.item())
                
                self.optimizer.step()
        
        self.memory.clear()
        n_batches = len(batches) * self.n_epochs
        self.last_policy_loss = total_policy_loss / n_batches
        self.last_value_loss = total_value_loss / n_batches
        self.last_entropy = total_entropy / n_batches
        self.last_total_loss = total_loss / n_batches
        self.last_clipped_updates = total_clipped / n_batches
        self.last_update_ratio = total_update_ratio / n_batches
    
    def _compute_gae(self, states, values, rewards, dones):
        self.actor_critic.eval()
        with torch.no_grad():
            _, last_value = self.actor_critic(states[-1])
        last_value = last_value.item()
        
        advantages = np.zeros(len(rewards), dtype=np.float32)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            # If t is the last step, use the last value
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # Compute GAE
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages[t] = gae
        
        return advantages
    
    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
    
    def logs(self):
        print(f"Update interval: {self.update_interval}, Current steps: {self.step_counter}")
        if hasattr(self, 'last_total_loss'):
            print(f"Last total loss: {self.last_total_loss:.4f}")
        if hasattr(self, 'last_policy_loss'):
            print(f"Last policy loss: {self.last_policy_loss:.4f}")
        if hasattr(self, 'last_value_loss'):
            print(f"Last value loss: {self.last_value_loss:.4f}")
        if hasattr(self, 'last_entropy'):
            print(f"Last entropy: {self.last_entropy:.4f}")
        if hasattr(self, 'last_clipped_updates'):
            print(f"Last clipped updates: {self.last_clipped_updates}")
        if hasattr(self, 'last_update_ratio'):
            print(f"Last update ratio: {self.last_update_ratio:.4f}")
    
    def set_metrics_tracker(self, metrics_tracker):
        self.metrics_tracker = metrics_tracker