# inference.py
import torch
import torch.nn as nn
import os
import cv2
import random
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import vit_b_16 as ViT, ViT_B_16_Weights
from skimage import io

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best fine-tuned model
best_model_path = "saved/ViT_Finetune/best_finetuned_Adam_lr5e-06_epoch_9.pth"  # Adjust based on the best-performing model
model = ViT(weights=ViT_B_16_Weights.DEFAULT)

# Modify classifier head
model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=100, bias=True))

# Move model to GPU(s)
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    model = nn.DataParallel(model)
model.to(device)

# Load fine-tuned weights
if os.path.exists(best_model_path):
    print(f"Loading best fine-tuned model: {best_model_path}")
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)
else:
    raise FileNotFoundError(f"Best model not found at {best_model_path}")

# Set model to evaluation mode
model.eval()

# Image directories
test_dir = "../data/test"
predict_dir = "./sport_dataset"

# Image transformation (same as training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load class names
class_file = "../data/sports.csv"
class_df = pd.read_csv(class_file, usecols=['class id', 'labels'])
class_dict = {row[0]: row[1] for _, row in class_df.iterrows()}

# Function to perform inference on a single image
def predict_image(image_path, model, transform, device):
    image = cv2.imread(image_path)

    if image.shape[-1] != 3:  # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    
    return class_dict[predicted_class.item()]

# Perform inference on test images
test_images = sorted(os.listdir(test_dir))[:10]  # Get first 10 images
predict_images = sorted(os.listdir(predict_dir))[:10]  # Get first 10 images

test_images = [os.path.join(test_dir, folder, img) for folder in test_images for img in os.listdir(os.path.join(test_dir, folder))]

test_results = [(img, predict_image(img, model, transform, device)) for img in test_images]
predict_results = [(img, predict_image(os.path.join(predict_dir, img), model, transform, device)) for img in predict_images]

# Function to visualize predictions
def visualize_predictions(image_dir, results, title):
    plt.figure(figsize=(12, 6))
    for i, (img_name, pred) in enumerate(random.sample(results, min(5, len(results)))):  # Random 5 images
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # if image.shape[-1] != 3:  # Convert grayscale to RGB
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        plt.subplot(1, 5, i + 1)
        plt.imshow(image)
        plt.title(f"Pred: {pred}", fontsize=10)
        plt.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.savefig(f'figs/{title}.jpg')
    plt.show()

# Visualize test predictions
visualize_predictions('', test_results, "Test Image Predictions")

# Visualize 'images to predict' predictions
visualize_predictions(predict_dir, predict_results, "Images to Predict - Predictions")
