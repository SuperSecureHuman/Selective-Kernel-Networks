import timm
import torch
import torch.nn as nn
from torchsummary import summary

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


model = timm.create_model('skresnet18', pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-2]))

# new model --> SE Block --> Global Average Pooling --> Linear

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        self.model = model
        self.se = SE_Block(512)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512, 4)

    def forward(self, x):
        x = self.model(x)
        x = self.se(x)
        x = self.avg_pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

model = FinalModel()


def return_model():
    return model