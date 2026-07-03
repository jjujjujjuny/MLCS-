#==========================================#
# Title:  cnn network
# Author: Jaewoong Han
# Date:   2025-06-27
#==========================================#
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, batch_size):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.layer = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # [100,3,224,224] -> [100,2048]
        for param in self.layer.parameters():
            param.requires_grad = False
        num_ftrs = self.layer.fc.in_features
        self.layer.fc = nn.Identity()

        self.fc_layer = nn.Sequential(
            nn.Linear(num_ftrs,100), # [100,2048] -> [100,100]
            nn.ReLU(),
            nn.Linear(100,10) # [100,100] -> [100,10]
        )

    def forward(self,x):
        out = self.layer(x)
        out = out.view(self.batch_size,-1)
        out = self.fc_layer(out)
        return out