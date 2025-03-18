import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from .network_unified import QNetworkUnified
from .preprocessing import ImagePreprocessor

class AgentDoubleDQN:
    def __init__(
        self,
        state_size,
        action_size,
        seed=1993,
        nb_hidden=(256, 128),
        learning_rate=0.0001,
        memory_size=10000,
        batch_size=32,
        gamma=0.99,
        tau=0.001,
        update_every=4,
        epsilon_enabled=True,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99995,
        model_dir="../models/DQN_double.pt",
        feature_extractor='resnet',
        finetune_features=False,
        use_frame_stack=True,
        frame_stack_size=4,
        target_size=(84, 84),
        preprocess_method="enhanced"
    ):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}", flush=True)
        print(f"Using Double DQN algorithm", flush=True)
        
        # Image feature related parameters
        self.feature_extractor = feature_extractor
        self.finetune_features = finetune_features
        self.use_frame_stack = use_frame_stack
        self.frame_stack_size = frame_stack_size
        self.target_size = target_size
        self.preprocess_method = preprocess_method
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(
            method=preprocess_method,
            target_size=target_size,
            use_frame_stack=use_frame_stack,
            frame_stack_size=frame_stack_size,
        )
        
        
        # Q-Network
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
        
        # Set up optimizer with different learning rates for feature extractor and Q-value heads
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
            # Standard optimizer setup
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
            if feature_extractor != 'spatial_cnn':
                print(f"Using frozen {feature_extractor} feature extractor", flush=True)
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.UPDATE_EVERY = update_every
        self.GAMMA = gamma
        self.TAU = tau
        
        # Epsilon-greedy action selection
        self.epsilon_enabled = epsilon_enabled
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.Experience = namedtuple("Experience", 
                                   ["state", "action", "reward", "next_state", "done"])
        
        self.model_dir = model_dir

    def step(self, state, action, reward, next_state, done):
        # Process the state and next_state if they are raw RGB images
        if isinstance(state, np.ndarray) and state.ndim == 3 and state.shape[2] == 3:
            # If state is a raw RGB image
            state = self.preprocessor.process(state)
            
        if isinstance(next_state, np.ndarray) and next_state.ndim == 3 and next_state.shape[2] == 3:
            # If next_state is a raw RGB image
            next_state = self.preprocessor.process(next_state)
        
        # Save experience in replay memory
        self.memory.append(self.Experience(state, action, reward, next_state, done))
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self._sample_experiences()
            self._learn(experiences)

    def act(self, state):
        """Returns actions for given state as per current policy"""
        # Process the state if it's a raw RGB image
        if isinstance(state, np.ndarray) and state.ndim == 3 and state.shape[2] == 3:
            # If state is a raw RGB image
            
            state = self.preprocessor.process(state)
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if self.epsilon_enabled and random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def _learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        
        This is the key method that implements Double DQN.
        """
        states, actions, rewards, next_states, dones = experiences

        # Double DQN: Use local network to select actions and target network to evaluate them
        # 1. Get the actions that would be selected by the local network
        self.qnetwork_local.eval()
        with torch.no_grad():
            local_actions = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)
        self.qnetwork_local.train()
        
        # 2. Use the target network to evaluate the Q-values of those actions
        next_q_values = self.qnetwork_target(next_states).gather(1, local_actions)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.GAMMA * next_q_values * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute TD error for metrics
        td_error = (Q_targets - Q_expected).abs().mean().item()
        
        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        loss_value = loss.item()
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # Calculate gradient norm for metrics
        grad_norm = 0
        for param in self.qnetwork_local.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Track metrics if metrics_tracker is available
        if hasattr(self, 'metrics_tracker') and self.metrics_tracker is not None:
            self.metrics_tracker.log_loss(total=loss_value, td_error=td_error)
            self.metrics_tracker.log_param_stats(self.qnetwork_local, grad_norm=grad_norm)
        
        # Store last loss and TD error for logging
        self.last_loss = loss_value
        self.last_td_error = td_error
        
        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target)
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

    def _sample_experiences(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Ensure proper shape for image inputs
        states = np.stack([e.state for e in experiences])
        next_states = np.stack([e.next_state for e in experiences])
        
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)

    def save_model(self, path=None):
        """Save model to file"""
        save_path = path if path is not None else self.model_dir
        try:
            torch.save(self.qnetwork_local.state_dict(), save_path)
            print(f"Model saved to {save_path}", flush=True)
        except Exception as e:
            print(f"Error saving model: {e}", flush=True)

    def load_model(self, path=None, device=None):
        """Load model from file
        
        Args:
            path (str): Path to the saved model file
            device (str or torch.device): Device to load the model to ('cuda', 'cpu', or torch.device)
                                         If None, will try to use CUDA if available, else CPU
        """
        load_path = path if path is not None else self.model_dir
        
        # Determine device to load model to
        if device is None:
            # Try to use CUDA if available, otherwise use CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            # Convert string to torch.device
            device = torch.device(device)
        
        try:
            # Load model with specified device
            self.qnetwork_local.load_state_dict(torch.load(load_path, map_location=device))
            self.qnetwork_target.load_state_dict(torch.load(load_path, map_location=device))
            print(f"Model loaded from {load_path} to {device}", flush=True)
        except Exception as e:
            print(f"Error loading model: {e}", flush=True)
            # Attempt fallback to CPU if loading to GPU failed
            if device.type == 'cuda':
                try:
                    print("Attempting fallback to CPU...", flush=True)
                    cpu_device = torch.device('cpu')
                    self.qnetwork_local.load_state_dict(torch.load(load_path, map_location=cpu_device))
                    self.qnetwork_target.load_state_dict(torch.load(load_path, map_location=cpu_device))
                    print(f"Model successfully loaded to CPU instead", flush=True)
                except Exception as fallback_error:
                    print(f"CPU fallback also failed: {fallback_error}", flush=True)

    def logs(self):
        """Return logs for the agent"""
        log_data = {
            "epsilon": self.epsilon,
            "algorithm": "Double DQN",
            "feature_extractor": self.feature_extractor,
            "finetune_features": self.finetune_features
        }
        
        # Add loss and TD error if available
        if hasattr(self, 'last_loss'):
            log_data["loss"] = self.last_loss
        if hasattr(self, 'last_td_error'):
            log_data["td_error"] = self.last_td_error
            
        return log_data
    
    def set_metrics_tracker(self, metrics_tracker):
        """Set metrics tracker for detailed logging during training"""
        self.metrics_tracker = metrics_tracker
    
    def reset(self):
        """Reset the preprocessor state"""
        if hasattr(self, 'preprocessor'):
            self.preprocessor.reset() 