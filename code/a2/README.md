# YOLOv4 Implementation

This project implements YOLOv4 object detection with Complete IoU (CIoU) loss function and evaluates its performance on the COCO dataset. The implementation includes whole pipeline for training and detection (inference)+CIoU loss integration.

## Training results
Please consider `logs` directory to see the results of training. Training was based on see the workage of pipeline, and trained in a single epoch based on time limitations.

## Usage
### Dependencies
```python
pip install -r requirements.txt
```

### Training
```python
python train.py --data ./data/coco.yaml --batch-size 4 --weights checkpoints/yolov4.weights --device 1
```

### Inferencing
```python
python detect.py --data data/coco.yaml --weights checkpoints/yolov4.weights --img cocoimages/000000521540.jpg --device 1
```

## Results

[1](https://github.com/eracoding/rtml/tree/main/code/a2/results/result_000000116031.jpg)
[2](https://github.com/eracoding/rtml/tree/main/code/a2/results/result_000000233141.jpg)
[3](https://github.com/eracoding/rtml/tree/main/code/a2/results/result_000000523923.jpg)
[4](https://github.com/eracoding/rtml/tree/main/code/a2/results/test.jpg)
