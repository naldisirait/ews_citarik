import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import torch
import torch.nn as nn

# Define a basic ResNet block
class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # Shortcut connection
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        
        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        return out

# Define the main model
class RegressionModelWithResNet(nn.Module):
    def __init__(self):
        super(RegressionModelWithResNet, self).__init__()
        # Initial fully connected layer
        self.fc_input = nn.Linear(144, 128)  # Reduce dimensionality before ResNet block
        
        # ResNet block
        self.resnet_block = BasicBlock(128, 128)
        
        # Output layer
        self.fc_output = nn.Linear(128, 168)
    
    def forward(self, x):
        x = self.fc_input(x)
        x = self.resnet_block(x)
        x = self.fc_output(x)
        return x

# Create an instance of the model
model = RegressionModelWithResNet()

# Summary of the model
print(model)

#