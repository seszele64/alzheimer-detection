"""
This code defines a custom neural network called AlzheimerNet. It uses a pre-trained EfficientNet-B0 model as the base and modifies the classifier layer to output the desired number of classes (4 in this case, for different stages of Alzheimer's). The forward method defines how input data flows through the network
"""

import torch.nn as nn
from torchvision import models

class AlzheimerNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerNet, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
