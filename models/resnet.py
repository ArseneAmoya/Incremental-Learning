from torchvision.models import resnet18
import torch.nn as nn
import torch

class ResNet18Cut(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = False):
        super(ResNet18Cut, self).__init__()
        resnet = resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten(1))  # Remove the last fully connected layer
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return torch.sigmoid(x)