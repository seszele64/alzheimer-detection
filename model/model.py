import torch.nn as nn
import torchvision.models as models

class AlzheimerNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerNet, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
