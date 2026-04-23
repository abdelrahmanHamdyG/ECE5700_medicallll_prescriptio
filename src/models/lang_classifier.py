
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class LanguageClassifier(nn.Module):
    """
    Language classifier for Arabic vs English.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(LanguageClassifier, self).__init__()
        
        self.model=mobilenet_v3_small(pretrained=pretrained)
        in_features=self.model.classifier[3].in_features


        # self.model.classifier[2] = nn.Dropout(p=0.25, inplace=True)

    
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

