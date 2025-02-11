# ViT Implementation from Scratch and Finetuning

This project implements ViT fine-tuning on classification task in custom [sports dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification). 

The objective of laboraty session was to apply fine-tuning, improved techniques, and usage of tensorboard which is efficient tool. The lab session was very informative.

## Training results
Please consider `logs` directory to see the results of training. Training was based on see the workage of pipeline, and trained in a single epoch based on time limitations. I have used multiple gpu training (precisely 4 gpus).

Based on obtained result {'lr': 5e-06, 'optimizer': 'Adam', 'batch_size': 256}. Best Accuracy: 96.4000% performing the best

| Optimizer | Batch_size | Learning Rate | Fine-tuned Epochs | Best Accuracy on Validation Set |
|-----------|------------|---------------|-------------------|---------------------------------|
| Adam      |   256      | 5e-06         |  9                | 96.4%                           |
| SGD       |   256      | 1e-5          |  55               | 93.8%                           |
| AdamW     |   256      | 5e-06         |  20               | 95.8%                           |


### Training gpu consumption based on batch_size=256:
```cmd
Tue Feb 11 14:53:25 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     Off |   00000000:84:00.0 Off |                  N/A |
| 47%   80C    P2            238W /  250W |   10962MiB /  11264MiB |     90%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 2080 Ti     Off |   00000000:85:00.0 Off |                  N/A |
| 46%   78C    P2            192W /  250W |    9182MiB /  11264MiB |     90%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 2080 Ti     Off |   00000000:88:00.0 Off |                  N/A |
| 50%   85C    P2            237W /  250W |    9720MiB /  11264MiB |     91%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 2080 Ti     Off |   00000000:89:00.0 Off |                  N/A |
| 51%   86C    P2            233W /  250W |    9824MiB /  11264MiB |     91%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A   1671596      C   /opt/tljh/user/bin/python                   10958MiB |
|    1   N/A  N/A   1671596      C   /opt/tljh/user/bin/python                    9178MiB |
|    2   N/A  N/A   1671596      C   /opt/tljh/user/bin/python                    9716MiB |
|    3   N/A  N/A   1671596      C   /opt/tljh/user/bin/python                    9820MiB |
+-----------------------------------------------------------------------------------------+
```

## Usage

### Training
```python
python finetune.py
```

### Inferencing
```python
python inference.py
```

## Results

![1](https://raw.githubusercontent.com/eracoding/rtml/main/code/a3/figs/acc1.png)
![2](https://raw.githubusercontent.com/eracoding/rtml/main/code/a3/figs/loss1.png)
![3](https://raw.githubusercontent.com/eracoding/rtml/main/code/a3/figs/acc2.png)
![4](https://raw.githubusercontent.com/eracoding/rtml/main/code/a3/figs/loss2.png)