import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from .networks import ActorCritic
from .preprocessing import ImagePreprocessor

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
               np.array(self.probs), np.array(self.vals), batches

class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        seed=1993,
        nb_hidden=(256, 128),
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_interval=2048,
        model_dir="../models/PPO_RGB.pt",
        feature_extractor="resnet",  # Fixed to ResNet
        finetune_features=False,
        use_frame_stack=True,
        frame_stack_size=4,
        target_size=(84, 84),
        preprocess_method="basic"
    ):
        
        self.state_size = state_size  
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_interval = update_interval
        self.model_dir = model_dir
        
        self.feature_extractor = feature_extractor 
        self.finetune_features = finetune_features
        self.use_frame_stack = use_frame_stack
        self.frame_stack_size = frame_stack_size
        self.target_size = target_size
        self.preprocess_method = preprocess_method
        
        self.seed = torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.preprocessor = ImagePreprocessor(
            method=preprocess_method,
            target_size=target_size,
            use_frame_stack=use_frame_stack,
            frame_stack_size=frame_stack_size,
        )
        
        # Actor-Critic network
        self.actor_critic = ActorCritic(
            state_size=state_size,
            action_size=action_size,
            feature_extractor=feature_extractor,
            hidden_size=nb_hidden
        ).to(self.device)
        
        if finetune_features:
            feature_params = list(self.actor_critic.features.parameters())
            head_params = list(self.actor_critic.actor.parameters()) + list(self.actor_critic.critic.parameters())
            
            self.optimizer = optim.Adam([
                {'params': feature_params, 'lr': learning_rate * 0.1},  # Lower LR for pre-trained features
                {'params': head_params, 'lr': learning_rate}
            ])
            print(f"Fine-tuning ResNet with reduced learning rate")
        else:
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
            print("Using frozen ResNet feature extractor")
        
        self.memory = PPOMemory(batch_size)
        
        self.timesteps_since_learn = 0
        self.last_loss = 0
        self.last_value_loss = 0
        self.last_policy_loss = 0
        self.last_entropy = 0
        
        self.metrics_tracker = None
    
    def act(self, state):
        if isinstance(state, np.ndarray) and state.ndim == 3 and state.shape[2] == 3:
            state = self.preprocessor.process(state)
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.actor_critic.eval()
        with torch.no_grad():
            policy, value = self.actor_critic(state)
            
            dist = Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            entropy = dist.entropy().item()
            
            if hasattr(self, 'metrics_tracker') and self.metrics_tracker is not None:
                self.metrics_tracker.log_exploration(entropy)
            
            return action.item(), log_prob.item(), value.item()
    
    def evaluate(self, states, actions):
        """Evaluate actions using current policy and critic"""
        self.actor_critic.eval()
        
        policy, values = self.actor_critic(states)
        
        # Get log probabilities
        dist = Categorical(policy)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return action_log_probs, values, entropy
    
    def step(self, state, action, reward, next_state, done):
        """Store transition in memory and learn if it's time"""
        if isinstance(state, np.ndarray) and state.ndim == 3 and state.shape[2] == 3:
            processed_state = self.preprocessor.process(state)
        else:
            processed_state = state
        
        state_tensor = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, value = self.actor_critic(state_tensor)
            
        action_prob = policy[0, action].item()
        value = value.item()
        
        self.memory.store(processed_state, action, action_prob, value, reward, done)
        
        self.timesteps_since_learn += 1
        
        if self.timesteps_since_learn >= self.update_interval:
            self.learn()
            self.timesteps_since_learn = 0
            
            if done and self.use_frame_stack:
                self.preprocessor.reset()
            
            return True
        
        if done and self.use_frame_stack:
            self.preprocessor.reset()
            
        return False
    
    def learn(self):
        """Update actor and critic networks using PPO"""
        states, actions, old_probs, values, batches = self.memory.generate_batches()
        
        advantages = self._compute_advantages(
            rewards=np.array(self.memory.rewards),
            values=values,
            dones=np.array(self.memory.dones)
        )
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = advantages + values
        
        self.actor_critic.train()
        
        total_loss_sum = 0
        actor_loss_sum = 0
        critic_loss_sum = 0
        entropy_sum = 0
        total_clipped = 0
        total_update_ratio = 0
        
        for _ in range(self.n_epochs):
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_probs = old_probs[batch]
                batch_returns = returns[batch]
                batch_advantages = advantages[batch]
                
                action_log_probs, critic_values, entropy = self.evaluate(
                    batch_states, batch_actions
                )
                
                # Calculate the probability ratio
                ratios = torch.exp(action_log_probs - torch.log(batch_old_probs + 1e-10))
                
                # Count clipped updates for metrics
                clipped = (ratios < 1.0 - self.policy_clip).sum().item() + (ratios > 1.0 + self.policy_clip).sum().item()
                total_clipped += clipped
                
                # Calculate average update ratio for metrics
                update_ratio = torch.abs(ratios - 1.0).mean().item()
                total_update_ratio += update_ratio
                
                # Calculate surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * batch_advantages
                
                # Calculate actor and critic losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(critic_values.squeeze(), batch_returns)
                entropy_loss = entropy.mean()
                
                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                
                grad_norm = 0
                for param in self.actor_critic.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                
                if self.metrics_tracker is not None:
                    self.metrics_tracker.log_loss(
                        total=total_loss.item(),
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
                
                total_loss_sum += total_loss.item()
                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                entropy_sum += entropy_loss.item()
        
        n_updates = len(batches) * self.n_epochs
        self.last_loss = total_loss_sum / n_updates
        self.last_policy_loss = actor_loss_sum / n_updates
        self.last_value_loss = critic_loss_sum / n_updates
        self.last_entropy = entropy_sum / n_updates
        
        # Clear memory after update
        self.memory.clear()
    
    def _compute_advantages(self, rewards, values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # Add dummy value for terminal states
        values = np.append(values, 0.0)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages[t] = gae
        
        return advantages
    
    def save_model(self, path=None):
        if path is None:
            path = self.model_dir

        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_extractor': self.feature_extractor,
            'preprocess_method': self.preprocess_method,
            'state_size': self.state_size,
            'action_size': self.action_size,
        }, path)
    
    def load_model(self, path=None):
        if path is None:
            path = self.model_dir
        
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'feature_extractor' in checkpoint:
            self.feature_extractor = checkpoint['feature_extractor']
        if 'preprocess_method' in checkpoint:
            self.preprocess_method = checkpoint['preprocess_method']
    
    def logs(self):
        return {
            "feature_extractor": self.feature_extractor,
            "preprocess_method": self.preprocess_method,
            "finetune": "yes" if self.finetune_features else "no",
            "loss": self.last_loss,
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy
        }
        
    def set_metrics_tracker(self, metrics_tracker):
        self.metrics_tracker = metrics_tracker