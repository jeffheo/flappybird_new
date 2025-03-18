import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch.hub

class SpatialAwareCNN(nn.Module):
    def __init__(self, input_channels=1):
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        return x

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, freeze_weights=True):
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.feature_dim = self.resnet.fc.in_features  # 512 for ResNet18
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        if freeze_weights:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        return self.resnet(x)


class QNetworkUnified(nn.Module):

    def __init__(self, state_size, action_size, feature_extractor='resnet', hidden_size=(256, 128), freeze_features=True):
       
        super(QNetworkUnified, self).__init__()
        self.feature_extractor_type = feature_extractor
        
        if isinstance(state_size, tuple) and len(state_size) == 3:
            input_channels = state_size[0]  
        else:
            input_channels = 1 
            
        print(f"Input channels for feature extractor: {input_channels}", flush=True)
        
        if feature_extractor == 'spatial_cnn':
            self.features = SpatialAwareCNN(input_channels=input_channels)
            feature_size = self.features.feature_dim
        elif feature_extractor == 'resnet':
            self.features = ResNetFeatureExtractor(input_channels=input_channels, freeze_weights=freeze_features)
            feature_size = self.features.feature_dim
        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor}")
        
        # Q-value network
        self.fc1 = nn.Linear(feature_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], action_size)

    def forward(self, x):
        x = self.features(x)
        
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) 