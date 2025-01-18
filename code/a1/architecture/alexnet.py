import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.custom_layers import LocalResponseNormalize

class AlexNet(nn.Module):
    def __init__(self, n_classes: int = 10):
        super(AlexNet, self).__init__()

        self.feature_extractors = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNormalize(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNormalize(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        x = self.feature_extractors(x)
        x = self.adaptive_pool(x)
        x = self.classification(x)

        return x


# Pure imitation to alexnet including GPU split
class VanilaAlexNet(nn.Module):
    """
    This is a pure implementation of AlexNet architecture following the GPU immitation. To train on two gpus, we would need to convert input image into
    corresponding gpu.
    """
    def __init__(self):
        super(AlexNet, self).__init__()

        # GPU1
        self.conv1_1 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
        self.maxpool1_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Response normalization layer
        self.local_normalization_1_1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        
        self.conv2_1 = nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2)
        self.maxpool2_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Response normalization layer
        self.local_normalization_1_2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)

        self.conv3_1 = nn.Conv2d(256, 192, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.maxpool3_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1_1 = nn.Linear(6 * 6 * 256, 2048)
        self.fc2_1 = nn.Linear(4096, 2048)
        
        # GPU2
        self.conv1_2 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
        self.maxpool1_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Response normalization layer
        self.local_normalization_2_1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        
        self.conv2_2 = nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2)
        self.maxpool2_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Response normalization layer
        self.local_normalization_2_2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)

        self.conv3_2 = nn.Conv2d(256, 192, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.maxpool3_2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1_2 = nn.Linear(6 * 6 * 256, 2048)
        self.fc2_2 = nn.Linear(4096, 2048)

        # combine
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(4096, 1000)

    def forward(self, img, gpu1="cuda:0", gpu2="cuda:1"):
        # GPU 1
        x1 = self.maxpool1_1(self.local_normalization_1_1(F.relu(self.conv1_1(img))))

        x1 = self.maxpool2_1(self.local_normalization_1_2(F.relu(self.conv2_1(x1))))

        # GPU 2
        x2 = self.maxpool1_2(self.local_normalization_2_1(F.relu(self.conv1_2(img))))

        x2 = self.maxpool2_2(self.local_normalization_2_2(F.relu(self.conv2_2(x2))))

        x = torch.cat([x1, x2])

        x1 = F.relu(self.conv3_1(x))
        x1 = F.relu(self.conv4_1(x1))
        x1 = self.maxpool3_1(F.relu(self.conv5_1(x1)))

        x2 = F.relu(self.conv3_2(x))
        x2 = F.relu(self.conv4_2(x2))
        x2 = self.maxpool3_2(F.relu(self.conv5_2(x2)))

        x = torch.flatten(torch.cat([x1, x2]))

        x1 = self.fc1_1(x)
        x1 = self.dropout(x1)
        x2 = self.fc1_2(x)
        x2 = self.dropout(x2)

        x = torch.cat([x1, x2])

        x1 = self.fc2_1(x)
        x1 = self.dropout(x1)
        x2 = self.fc2_2(x)
        x2 = self.dropout(x2)

        x = torch.cat([x1, x2])

        x = self.fc(x)

        return x

def sequential_alexnet():
    class Flatten(nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return x.view(batch_size, -1)

    # AlexNet-like model using the Sequential API

    NUM_CLASSES = 10

    alexnet_sequential = nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.AdaptiveAvgPool2d((6, 6)),
        Flatten(),
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, NUM_CLASSES)
    )

    return alexnet_sequential
