import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import torch.hub

def debug_tensor(tensor, name, print_stats=True):
    """Helper function to check for NaNs and print statistics"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf or print_stats:
        stats = {
            "name": name,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item()
        }
        print(f"DEBUG {name}: {stats}")
    
    return has_nan or has_inf

class SpatialAwareCNN(nn.Module):
    def __init__(self, input_channels=4):
        super(SpatialAwareCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 8))  
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.feature_dim = 256 * 4 * 8

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        return x.view(x.size(0), -1)  

class ResNetFeatureExtractor(nn.Module):
    """ResNet18 feature extractor with modifications for grayscale/stacked frames"""
    def __init__(self, input_channels=4):
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Output feature dimension
        self.feature_dim = 512

    def forward(self, x):
        x = self.resnet(x)
        return x.view(x.size(0), -1)  


class ActorCritic(nn.Module):
    """Actor-Critic Network with flexible feature extraction"""
    def __init__(self, 
                 state_size, 
                 action_size, 
                 feature_extractor='spatial_cnn',
                 hidden_size=(256, 128)):
        super(ActorCritic, self).__init__()
        
        input_channels = state_size[0]  
        
        # Select feature extractor based on configuration
        if feature_extractor == 'spatial_cnn':
            self.features = SpatialAwareCNN(input_channels)
            feature_dim = self.features.feature_dim
        elif feature_extractor == 'resnet':
            self.features = ResNetFeatureExtractor(input_channels)
            feature_dim = self.features.feature_dim
        else:
            # Default to spatial aware CNN
            self.features = SpatialAwareCNN(input_channels)
            feature_dim = self.features.feature_dim
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], action_size)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1)
        )
    
    def forward(self, x):
        debug_tensor(x, "network_input")
        features = self.features(x)
        debug_tensor(features, "resnet_features")
        
        action_logits = self.actor(features)
        
        action_logits = torch.clamp(action_logits, -20.0, 20.0)
        
        policy = F.softmax(action_logits, dim=-1)
        debug_tensor(policy, "policy_probs")
        
        policy = policy + 1e-8
        policy = policy / policy.sum(dim=-1, keepdim=True)  
        
        value = self.critic(features)
        debug_tensor(value, "value")
        
        return policy, value