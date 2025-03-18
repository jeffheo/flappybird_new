import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from .network import QNetwork

class AgentDoubleDQNState:
    def __init__(
        self,
        state_size,
        action_size,
        seed=1993,
        nb_hidden=(64, 64),
        learning_rate=0.0005,
        memory_size=100000,
        prioritized_memory=False,
        batch_size=64,
        gamma=0.99,
        tau=0.001,
        small_eps=0.03,
        update_every=4,
        epsilon_enabled=True,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99995,
        model_dir="../models/DoubleDQN_state.pt",
        use_cnn=False
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_size=nb_hidden, use_cnn=use_cnn).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_size=nb_hidden, use_cnn=use_cnn).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        self.t_step = 0
        self.UPDATE_EVERY = update_every
        self.GAMMA = gamma
        self.TAU = tau
        
        self.epsilon_enabled = epsilon_enabled
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.prioritized_memory = prioritized_memory
        self.small_eps = small_eps
        
        self.model_dir = model_dir
        
        self.last_loss = 0
        self.last_td_error = 0
        self.metrics_tracker = None
        
        self.Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def step(self, state, action, reward, next_state, done):
        e = self.Experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self._sample_experiences()
            self._learn(experiences)
    
    def act(self, state):
        """Returns actions for given state using current policy."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        else:
            state = state.float().unsqueeze(0).to(self.device)
        
        if self.epsilon_enabled and random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        return np.argmax(action_values.cpu().data.numpy())
    
    def _learn(self, experiences):
        """This is for Double DQN."""
        states, actions, rewards, next_states, dones = experiences
        
        # 1. Get the actions that would be selected by the local network
        self.qnetwork_local.eval()
        with torch.no_grad():
            local_actions = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)
        self.qnetwork_local.train()
        
        # 2. Use the target network to evaluate the Q-values of those actions
        next_q_values = self.qnetwork_target(next_states).gather(1, local_actions)
        
        Q_targets = rewards + (self.GAMMA * next_q_values * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute TD error
        td_error = (Q_targets - Q_expected).abs().mean().item()
        self.last_td_error = td_error
        
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.last_loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norm = 0
        for param in self.qnetwork_local.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        if hasattr(self, 'metrics_tracker') and self.metrics_tracker is not None:
            self.metrics_tracker.log_loss(total=self.last_loss, td_error=self.last_td_error)
            self.metrics_tracker.log_param_stats(self.qnetwork_local, grad_norm=grad_norm)
        
        self.optimizer.step()
        
        self._soft_update(self.qnetwork_local, self.qnetwork_target)

        # Decay epsilon
        if self.epsilon_enabled:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
    
    def _sample_experiences(self):
        """Sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
    def save_model(self, path=None):
        if path is None:
            path = self.model_dir
        torch.save(self.qnetwork_local.state_dict(), path)
    
    def load_model(self, path=None):
        if path is None:
            path = self.model_dir
        self.qnetwork_local.load_state_dict(torch.load(path, map_location=self.device))
        self.qnetwork_target.load_state_dict(torch.load(path, map_location=self.device))
    
    def logs(self):
        print(f"Epsilon: {self.epsilon:.4f}")
        print(f"Memory size: {len(self.memory)}")
        if hasattr(self, 'last_loss'):
            print(f"Last loss: {self.last_loss:.4f}")
        if hasattr(self, 'last_td_error'):
            print(f"Last TD error: {self.last_td_error:.4f}")
    
    def set_metrics_tracker(self, metrics_tracker):
        """Set metrics tracker."""
        self.metrics_tracker = metrics_tracker 