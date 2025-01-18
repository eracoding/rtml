import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.inception import Inception


class GoogleNet(nn.Module):
    def __init__(self, n_classes: int = 1000):
        super(GoogleNet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)

        # auxilary classifier 1
        self.aux1_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.aux1_conv = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.aux1_bn = nn.BatchNorm2d(128)
        self.aux1_fc1 = nn.Linear(4 * 4 * 128, 1024)
        self.aux1_dropout = nn.Dropout(p=0.7) # 70% ratio
        self.aux1_fc2 = nn.Linear(1024, n_classes)

        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)

        # auxilary classifier 2
        self.aux2_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.aux2_conv = nn.Conv2d(528, 128, kernel_size=1, stride=1)
        self.aux2_bn = nn.BatchNorm2d(128)
        self.aux2_fc1 = nn.Linear(4 * 4 * 128, 1024)
        self.aux2_dropout = nn.Dropout(p=0.7) # 70% ratio
        self.aux2_fc2 = nn.Linear(1024, n_classes)

        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear  = nn.Linear(1024, n_classes)

    
    def forward(self, x, train=False):
        aux1, aux2 = 0, 0
        
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)

        if train:
            aux1 = self.aux1_avgpool(out)
            aux1 = F.relu(self.aux1_bn(self.aux1_conv(aux1)))
            aux1 = torch.flatten(aux1, 1)
            aux1 = self.aux1_dropout(F.relu(self.aux1_fc1(aux1)))
            aux1 = self.aux1_fc2(aux1)            

        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)

        if train:
            aux2 = self.aux2_avgpool(out)
            aux2 = F.relu(self.aux2_bn(self.aux2_conv(aux2)))
            aux2 = torch.flatten(aux2, 1)
            aux2 = self.aux2_dropout(F.relu(self.aux2_fc1(aux2)))
            aux2 = self.aux2_fc2(aux2)

        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        if train:
            return out, aux1, aux2

        return out
