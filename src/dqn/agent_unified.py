import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from .network_unified import QNetworkUnified
from .preprocessing import ImagePreprocessor

class AgentUnified:
    def __init__(
        self,
        state_size,
        action_size,
        seed=0,
        nb_hidden=(64,64),
        learning_rate=0.0005,
        memory_size=100000,
        batch_size=64,
        gamma=0.99,
        tau=0.001,
        update_every=4,
        epsilon_enabled=True,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99995,
        model_dir="../models/DQN_unified.pt",
        feature_extractor='resnet',
        finetune_features=False,
        use_frame_stack=True,
        frame_stack_size=4,
        target_size=(224, 224),
        preprocess_method="enhanced"
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}", flush=True)
        
        self.feature_extractor = feature_extractor
        self.finetune_features = finetune_features
        self.use_frame_stack = use_frame_stack
        self.frame_stack_size = frame_stack_size
        self.target_size = target_size
        self.preprocess_method = preprocess_method
        
        self.preprocessor = ImagePreprocessor(
            target_size=target_size,
        )
        
        self.qnetwork_local = QNetworkUnified(
            state_size, 
            action_size, 
            feature_extractor=feature_extractor,
            hidden_size=nb_hidden,
            freeze_features=not finetune_features  # Freeze features if not fine-tuning
        ).to(self.device)
        
        self.qnetwork_target = QNetworkUnified(
            state_size, 
            action_size, 
            feature_extractor=feature_extractor,
            hidden_size=nb_hidden,
            freeze_features=not finetune_features  # Freeze features if not fine-tuning
        ).to(self.device)
        
        if finetune_features:
            # If fine-tuning feature extractor, use lower learning rate for feature extractor
            feature_params = list(self.qnetwork_local.features.parameters())
            head_params = list(self.qnetwork_local.fc1.parameters()) + \
                          list(self.qnetwork_local.fc2.parameters()) + \
                          list(self.qnetwork_local.fc3.parameters())
            
            self.optimizer = optim.Adam([
                {'params': feature_params, 'lr': learning_rate * 0.1},  # Lower LR for pre-trained features
                {'params': head_params, 'lr': learning_rate}
            ])
            print(f"Fine-tuning {feature_extractor} with reduced learning rate", flush=True)
        else:
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
            print(f"Using frozen {feature_extractor} feature extractor", flush=True)
        
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        self.t_step = 0
        self.UPDATE_EVERY = update_every
        self.GAMMA = gamma
        self.TAU = tau
        
        self.epsilon_enabled = epsilon_enabled
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.Experience = namedtuple("Experience", 
                                   ["state", "action", "reward", "next_state", "done"])
        
        self.model_dir = model_dir

    def step(self, state, action, reward, next_state, done):
        if isinstance(state, np.ndarray) and state.ndim == 3 and state.shape[2] == 3:
            
            state = self.preprocessor.process(state)
            
        if isinstance(next_state, np.ndarray) and next_state.ndim == 3 and next_state.shape[2] == 3:
            
            next_state = self.preprocessor.process(next_state)
        
        self.memory.append(self.Experience(state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self._sample_experiences()
            self._learn(experiences)

    def act(self, state):
        if isinstance(state, np.ndarray) and state.ndim == 3 and state.shape[2] == 3:
            
            state = self.preprocessor.process(state)
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if self.epsilon_enabled and random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def _learn(self, experiences):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        td_error = (Q_targets - Q_expected).abs().mean().item()
        
        loss = nn.MSELoss()(Q_expected, Q_targets)
        loss_value = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norm = 0
        for param in self.qnetwork_local.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        if hasattr(self, 'metrics_tracker') and self.metrics_tracker is not None:
            self.metrics_tracker.log_loss(total=loss_value, td_error=td_error)
            self.metrics_tracker.log_param_stats(self.qnetwork_local, grad_norm=grad_norm)
        
        self.last_loss = loss_value
        self.last_td_error = td_error
        
        self.optimizer.step()

        self._soft_update(self.qnetwork_local, self.qnetwork_target)
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

    def _sample_experiences(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # proper shape for image inputs
        states = np.stack([e.state for e in experiences])
        next_states = np.stack([e.next_state for e in experiences])
        
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)

    def save_model(self, path=None):
        save_path = path if path is not None else self.model_dir
        try:
            torch.save(self.qnetwork_local.state_dict(), save_path)
            print(f"Model saved to {save_path}", flush=True)
        except Exception as e:
            print(f"Error saving model: {e}", flush=True)

    def load_model(self, path=None, device=None):
        
        load_path = path if path is not None else self.model_dir
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        try:
            self.qnetwork_local.load_state_dict(torch.load(load_path, map_location=device))
            self.qnetwork_target.load_state_dict(torch.load(load_path, map_location=device))
            print(f"Model loaded from {load_path} to {device}", flush=True)
        except Exception as e:
            print(f"Error loading model: {e}", flush=True)
            if device.type == 'cuda':
                try:
                    print("Attempting CPU.", flush=True)
                    cpu_device = torch.device('cpu')
                    self.qnetwork_local.load_state_dict(torch.load(load_path, map_location=cpu_device))
                    self.qnetwork_target.load_state_dict(torch.load(load_path, map_location=cpu_device))
                    print(f"Model successfully loaded to CPU instead", flush=True)
                except Exception as fallback_error:
                    print(f"CPU also failed: {fallback_error}", flush=True)

    def logs(self):
        log_data = {"epsilon": self.epsilon}
        
        if hasattr(self, 'last_loss'):
            log_data["loss"] = self.last_loss
        if hasattr(self, 'last_td_error'):
            log_data["td_error"] = self.last_td_error
            
        return log_data
        
    def set_metrics_tracker(self, metrics_tracker):
        self.metrics_tracker = metrics_tracker
        
    def reset(self):
        if hasattr(self, 'preprocessor'):
            self.preprocessor.reset()