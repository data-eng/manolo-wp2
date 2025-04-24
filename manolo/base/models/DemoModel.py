import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model
class DemoModel(nn.Module):
    def __init__(self, num_classes=10):
        super(DemoModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc1 = nn.Linear(128, 256)  # Input size matches pooled feature size
        self.fc2 = nn.Linear(128, num_classes)   # Final output layer for logits

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        
        # Global average pooling
        embeddings = self.global_pool(x).view(x.size(0), -1)  # Shape: (batch_size, 128)
        
        # Final classification layer
        logits = self.fc2(embeddings)  # Shape: (batch_size, num_classes)
        
        return logits