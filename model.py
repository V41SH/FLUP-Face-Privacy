import torchvision
from torchvision.models import (resnet50, ResNet50_Weights,
                                resnet18, ResNet18_Weights,
                                mobilenet_v2, MobileNet_V2_Weights,
                                mobilenet_v3_large, MobileNet_V3_Large_Weights
                                )
import torch
import torch.nn as nn
import math

class SlayNet(nn.Module):

    def __init__(self, inputsize=(250), embedding_size=(512)):
        super().__init__()

        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2).features

        linear_input_size = 1280 * (math.ceil(inputsize/32) ** 2)
        # mobilenet was designed with 32x32 in mind. shouldn't be too much of an issue.

        self.linear = nn.Linear(linear_input_size, embedding_size)

    def forward(self, x):

        batchsize = x.shape[0]
        # print(x.shape)


        x = self.backbone(x)
        x = x.reshape(batchsize, -1) # flatten
        x = self.linear(x)

        return x
    
if __name__ == "__main__":
    model = SlayNet()
    print(model.backbone)

    inp = torch.rand((1,3,250,250))
    print(inp.shape)

    out = model(inp)
    print(out.shape)
