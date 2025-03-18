import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=(64, 64), use_cnn=False):
        super(QNetwork, self).__init__()
        self.use_cnn = use_cnn
        
        if use_cnn:
            self.conv1 = nn.Conv2d(state_size[0], 16, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)  # Additional conv layer
            
            sample_input = torch.zeros(1, *state_size)
            conv_out = self._forward_conv(sample_input)
            self.feature_size = int(np.prod(conv_out.shape[1:]))
            
            self.fc1 = nn.Linear(self.feature_size, 512)
            self.fc2 = nn.Linear(512, action_size)
        else:
            self.fc1 = nn.Linear(state_size, hidden_size[0])
            self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
            self.fc3 = nn.Linear(hidden_size[1], action_size)

    def _forward_conv(self, x):
        """Forward pass through convolutional layers."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        if self.use_cnn:
            x = self._forward_conv(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = F.relu(self.fc1(x))
            return self.fc2(x)
        else:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x) 