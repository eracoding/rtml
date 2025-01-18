# Model_zoo file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms, models
# from summary import summary
from summary import summary_string

from architecture.alexnet import AlexNet
from architecture.googlenet import GoogleNet
from utils import train_model, plot_data

torch.manual_seed(42)
# ADD config.yaml to control the hyperparameters and retrieve from there


def get_cifar10():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10('../data', download=True, train=True, transform=train_transform)
    test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=test_transform)

    train_data, val_data = torch.utils.data.random_split(train_dataset, [40000, 10000])

    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    return train_dataloader, val_dataloader, test_dataloader


def metrics(model, val_acc_history, loss_acc_history, filename="alexnet", input_model=(3, 224, 224)):
    plot_data(val_acc_history, loss_acc_history, model_name=filename)
    # summary(model, input_size=input_model) # input type dynamic
    
    model_summary, _ = summary_string(model, input_size=input_model)
    
    with open('{filename}_summary.txt', 'w') as f:
        f.write(str(model_summary))


def main():
    # TODO: Add argparse
    # if alexnet
    train_dataloader, val_dataloader, test_dataloader = get_cifar10()
    dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

    alexnet = AlexNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

    best_model, val_acc_history, loss_acc_history = train_model(alexnet, dataloaders, criterion, optimizer, device, 10, "main_alex_module_lr_0.001_best")

    metrics(best_model, val_acc_history, loss_acc_history, filename="alexnet")


    googlenet = GoogleNet(n_classes=10).to(device)
    criterion2 = nn.CrossEntropyLoss()
    optimizer2 = torch.optim.SGD(googlenet.parameters(), momentum=0.9, lr=0.001)

    best_model2, val_acc_history2, loss_acc_history2 = train_model(googlenet, dataloaders, criterion2, optimizer2, device, 15, "main_googlenet_module_lr_0.001_best", is_inception=True)

    metrics(best_model2, val_acc_history2, loss_acc_history2, filename="googlenet")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main()
