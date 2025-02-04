import torch
import cv2
import numpy as np
from model.darknet import Darknet
from utils.util import *
from model.yolov4 import YOLOv4Loss
from utils.metrics import evaluate_coco_map
from pycocotools.coco import COCO
import argparse
import os
import yaml
import math
from PIL import Image

class ImagePreprocessor:
    def __call__(self, img):
        img_array = np.asarray(img)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        img_array = cv2.resize(img_array, (608, 608))
        img_array = img_array.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.tensor(img_array)

class COCODataHandler:
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.img_ids = sorted(self.coco.imgs.keys())
        self.transform = transform
        
        self.label_map = {cat_id: idx for idx, cat_id in enumerate(self.coco.cats.keys())}
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        img_meta = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_meta['file_name'])
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        
        if self.transform:
            img = self.transform(img)
        
        boxes, labels = [], []
        for annotation in annotations:
            x, y, w, h = annotation['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.label_map[annotation['category_id']])
        
        target_data = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor(original_size)
        }
        
        return img, target_data
    
    def __len__(self):
        return len(self.img_ids)
    
    @staticmethod
    def collate_fn(data_batch):
        images, targets = zip(*data_batch)
        return torch.stack(images, 0), list(targets)

def train_yolo(model, train_loader, val_loader, coco_gt, device, num_epochs=100):
    """Train YOLOv4 with CIoU loss and mAP evaluation"""
    print("\nStarting training...")
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = YOLOv4Loss(model.anchors, num_classes=80)
    os.makedirs('checkpoints', exist_ok=True)
    
    best_map = 0.0
    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(device)
                for target in targets:
                    for key, val in target.items():
                        if isinstance(val, torch.Tensor):
                            target[key] = val.to(device)
                
                optimizer.zero_grad()
                predictions = model(images, device == torch.device("cuda:2"))
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}")
            
            print("\nEvaluating mAP...")
            model.eval()
            current_map = evaluate_coco_map(model, val_loader, coco_gt)
            print(f"Epoch {epoch} mAP: {current_map:.4f}")
            
            if current_map > best_map:
                best_map = current_map
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'mAP': current_map},
                           'checkpoints/yolov4_best.pth')
                print(f"Saved new best model with mAP: {current_map:.4f}")
        
        print("\nTraining complete!")
        print(f"Best mAP: {best_map:.4f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as error:
        print(f"\nError during training: {error}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='path to model config file')
    parser.add_argument('--weights', type=str, default='weights/yolov4.weights', help='path to weights file')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='path to data config file')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()
    
    print("Loading YOLOv4 Model...")
    model = Darknet(args.cfg)
    if os.path.exists(args.weights):
        model.load_weights(args.weights)
    else:
        print("No pretrained weights found. Training from scratch...")
    
    print("Loading data configuration...")
    with open(args.data, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    print(f"\nCreating datasets...")
    print(f"Training data: {data_dict['val']}")
    print(f"Training annotations: {data_dict['val_annotations']}")
    print(f"Validation data: {data_dict['val']}")
    print(f"Validation annotations: {data_dict['val_annotations']}")
    
    train_dataset = COCODataHandler(
        data_dict['val'],
        data_dict['val_annotations'],
        transform=ImagePreprocessor()
    )
    
    val_dataset = COCODataHandler(
        data_dict['val'],
        data_dict['val_annotations'],
        transform=ImagePreprocessor()
    )
    
    print(f"Dataset size: {len(train_dataset)} training images, {len(val_dataset)} validation images")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=COCODataHandler.collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=COCODataHandler.collate_fn
    )
    
    coco_gt = COCO(data_dict['val_annotations'])
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    train_yolo(model, train_loader, val_loader, coco_gt, device, num_epochs=args.epochs)

if __name__ == '__main__':
    main()
