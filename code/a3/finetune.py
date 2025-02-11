import errno
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from skimage import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.models import vit_b_16 as ViT, ViT_B_16_Weights

from tensorboardX import SummaryWriter


class SportDataset(Dataset):
    """Sport dataset."""

    def __init__(self, csv_file, root_dir, class_file, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            class_file (string): Path to the csv file with class names and indices.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        classes = pd.read_csv(class_file)
        self.class_dict = {row[2]:row[0] for i, row in classes.iterrows()}

        df = pd.read_csv(csv_file)
        df.drop(index=5621, inplace=True)
        
        if train:
            self.df = df[df['data set'] == 'train']
        else:
            self.df = df[df['data set'] == 'valid']

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 1])
        image = io.imread(img_name)

        if image.shape[-1] != 3:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        if self.transform:
            image = self.transform(image)

        label_keys = self.df.iloc[idx, 2]
        labels = self.class_dict[label_keys]
        labels = float(labels)

        sample = {'image': image, 'labels': labels}

        return sample

# Data Processing
train_transform = transforms.Compose([
    transforms.ToPILImage(mode="RGB"),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(mode="RGB"),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

csv_file = "../data/sports.csv"
class_file = "../data/sports.csv"
root_dir = "../data/"

train_ds = SportDataset(csv_file, root_dir, class_file, train=True, transform=train_transform)
val_ds = SportDataset(csv_file, root_dir, class_file, train=False, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# Helper function to display images
def show_images(images, labels, class_dict):
    plt.rcParams["figure.figsize"] = [15, 5]
    img_grid = torchvision.utils.make_grid(images)
    npimg = img_grid.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print(",".join([class_dict[label.item()] for label in labels]))


# Display a batch of images
dataiter = iter(train_loader)
batch = next(dataiter)
show_images(batch["image"], batch["labels"], pd.read_csv(class_file).set_index("class id")["labels"].to_dict())

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# model = ViT((1, 28, 28), n_patches=7, hidden_d=20, n_heads=2, out_d=10)
# model = model.to(device)


model = ViT(weights=ViT_B_16_Weights.DEFAULT)

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(model)
print(f"{total_params/1000000}M")

model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=100, bias=True))

# Set the GPU Device
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device)

# Move the model to Device
model.to(device)
print("Classifier Head: ", model.heads)
# Initiate the weights and biases
for m in model.heads:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=1e-6)


# Initialize Logger
class Logger:
    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name
        self.writer = SummaryWriter(comment=f"{model_name}_{data_name}")

    def log_loss(self, train_loss, val_loss, epoch):
        self.writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)

    def log_accuracy(self, train_acc, val_acc, epoch):
        self.writer.add_scalars("Accuracy", {"Train": train_acc, "Validation": val_acc}, epoch)

    def save_model(self, model, epoch, model_descr, val_acc):
        os.makedirs(f"saved/{self.model_name}", exist_ok=True)
        torch.save(model.state_dict(), f"saved/{self.model_name}/{model_descr}_epoch_{epoch}.pth")
        print(f"✅ New Best Model Saved at saved/{self.model_name}/{model_descr}_epoch_{epoch}.pth (Accuracy: {val_acc:.4%})")
        

    def close(self):
        self.writer.close()

# Experiment configurations
experiment_configs = [
    # {"lr": 5e-6, "optimizer": "Adam", "batch_size": 256},
    # {"lr": 1e-5, "optimizer": "SGD", "batch_size": 256},
    {"lr": 3e-6, "optimizer": "AdamW", "batch_size": 256},
]

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Best accuracy tracker
best_accuracy = 0.5
best_vloss = float('inf')
patience = 5

logger = Logger(model_name="ViT_Finetune", data_name="SportsDataset")

# Fine-Tuning Loop
for config in experiment_configs:
    # Initialize device, first we will move pretrained weights, then move our model to several gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the ViT model
    model = ViT(weights=ViT_B_16_Weights.DEFAULT)
    
    # Modify classifier head
    model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=100, bias=True))
    
    # Load checkpoint (7th epoch)
    file_name = 'Ep.7.pth'
    if os.path.exists(file_name):
        print(f"Loading checkpoint: {file_name}")
        state = torch.load(file_name, map_location=device)
        model.load_state_dict(state)
    
    # Lets move model to several gpus
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        
    # Move model to multiple GPUs
    if num_gpus > 1:
        model = nn.DataParallel(model)  # Enable multi-GPU training
    model.to(device)
    print(f"\n🔍 Starting Fine-Tuning with {config}\n")

    # Choose optimizer
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    # Update DataLoader with multi-GPU friendly batch size
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=num_gpus)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=num_gpus)

    additional_epochs = 100
    best_model_path = f"best_finetuned_{config['optimizer']}_lr{config['lr']}"
    epochs_without_improvement = 0

    for epoch in range(1, additional_epochs + 1):
        print(f"\n🚀 Fine-Tuning EPOCH {epoch}/{additional_epochs} | LR: {config['lr']}, Optimizer: {config['optimizer']}")

        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for data in tqdm(train_loader, desc='Training'):
            inputs, labels = data['image'].to(device), data['labels'].long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4%}")

        # Validation Phase
        model.eval()
        running_vloss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                inputs, labels = data['image'].to(device), data['labels'].long().to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                running_vloss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_vloss = running_vloss / len(val_loader)
        val_acc = correct_val / total_val
        print(f"Valid Loss: {avg_vloss:.4f}, Valid Accuracy: {val_acc:.4%}")

        # Log statistics
        logger.log_loss(avg_train_loss, avg_vloss, epoch)
        logger.log_accuracy(train_acc, val_acc, epoch)

        # Check for best accuracy improvement
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            logger.save_model(model, epoch, best_model_path, val_acc)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Check for early stopping
        if epochs_without_improvement >= patience:
            print(f"⏹ Early stopping triggered. No improvement in last {patience} epochs.")
            break

    print(f"\n🏆 Finished Fine-Tuning for {config}. Best Accuracy: {best_accuracy:.4%}")

# Close logger
logger.close()
