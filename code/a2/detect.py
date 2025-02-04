import argparse
import os

from model.darknet import Darknet
from utils.util import *

import torch
import cv2
import numpy as np
from pycocotools.coco import COCO
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
    def batch_collate(data_batch):
        images, targets = zip(*data_batch)
        return torch.stack(images, 0), list(targets)

def process_input_image(img_path, img_size=608):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image {img_path} not found")
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to load image: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0)

def draw_predictions(img, detections, class_labels=None):
    result_img = img.copy()
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
    
    for det in detections:
        _, x1, y1, x2, y2, obj_conf, cls_conf, cls_pred = det
        cls_pred = int(cls_pred)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = tuple(map(int, colors[cls_pred % len(colors)]))
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_labels[cls_pred] if class_labels else 'Class'} {cls_pred}: {cls_conf:.2f}"
        
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_img, (x1, y1 - label_size[1] - baseline - 5),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(result_img, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg')
    parser.add_argument('--weights', type=str, default='chechkpoints/yolov4.weights')
    parser.add_argument('--data', type=str, default='data/coco.yaml')
    parser.add_argument('--img', type=str, default='')
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()
    
    if args.img:
        print("Loading YOLOv4 Model...")
        model = Darknet(args.cfg)
        if os.path.exists(args.weights):
            model.load_weights(args.weights)
        else:
            print("No pretrained weights found. Training from scratch...")

        img_tensor = process_input_image(args.img)
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        model.to(device).eval()
        
        with torch.no_grad():
            predictions = model(img_tensor.to(device), device)
            detections = write_results(predictions, confidence=0.5, num_classes=80, nms_conf=0.4)
        
        if detections is not None:
            detections = detections.cpu().numpy()
            print(f"Detected {len(detections)} objects!")

            try:
                with open('data/coco.names', 'r') as f:
                    class_names = f.read().strip().split('\n')
            except:
                class_names = None
            
            for detection in detections:
                _, x1, y1, x2, y2, obj_conf, cls_conf, cls_pred = detection
                cls_pred = int(cls_pred)
                
                if class_names and cls_pred < len(class_names):
                    class_name = class_names[cls_pred]
                    print(f"Class: {class_name} ({cls_pred}), Confidence: {cls_conf:.4f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                else:
                    print(f"Class: {cls_pred}, Confidence: {cls_conf:.4f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
            result_img = draw_predictions(cv2.imread(args.img), detections)
            
            results_dir = os.path.join(os.getcwd(), 'results')
            os.makedirs(results_dir, exist_ok=True)

            orig_filename = os.path.basename(args.img)
            filename_without_ext = os.path.splitext(orig_filename)[0]
            output_path = os.path.join(results_dir, f'result_{filename_without_ext}.jpg')

            cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            print(f"\nDetection result saved to: {output_path}")
    else:
        print("Please specify image with: --img imageName")

if __name__ == '__main__':
    main()
