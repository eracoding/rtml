# Masked Auto-Encoder from Scratch and Transfer Learning on classification task

This project implements Masked Auto-Encoder from scratch training on MNIST to learn latent space. After pretraining, we downgrade to classification task for MNIST and Cifar-10 datasets. We will explore the performance, pipeline, and transfer learning implementations.

[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) is a self-supervised pretraining method for vision models, particularly Vision Transformers (ViTs). The key idea is to randomly mask a large portion of image patches, forcing the model to learn meaningful representations by reconstructing the missing parts.

Main Contributions:
- MAE outperforms contrastive learning-based approaches in multiple vision tasks.
- High masking ratio (75%), making the model learn semantically rich features.
- Better transferability to downstream tasks such as image classification, object detection, and segmentation.
- Masked Autoencoders (MAE) effectively learn image representations by reconstructing missing patches.


## Training resources
I have used puffer for training.

MAE training

![](https://raw.githubusercontent.com/eracoding/rtml/main/code/a4/media/mae_training.png)

MNIST classification training

![](https://raw.githubusercontent.com/eracoding/rtml/main/code/a4/media/mnist_training.png)

Cifar-10 classification training

![](https://raw.githubusercontent.com/eracoding/rtml/main/code/a4/media/cifar10_training.png)

## **Experiment Report: MNIST and CIFAR-10 Classification using ViT**
This report details the experiments conducted on MNIST and CIFAR-10 classification using a Vision Transformer (ViT) architecture. Initially, the model was trained on MNIST, achieving high accuracy. Subsequently, transfer learning was applied to CIFAR-10 by modifying the input structure, resulting in a significant drop in performance. Various optimizations are suggested to improve the CIFAR-10 results.

---

## **Experimental Setup**

#### **Model Architecture**
The ViT model consists of:
- **Patch Embedding Layer**: Converts input images into patches.
- **Transformer Encoder**: A stack of self-attention layers.
- **Classification Head**: A linear layer mapping embeddings to class logits.

Initially, the model was trained on MNIST with **image size = 28x28, patch size = 2**. Then I played with patch_size and changed it to be 4. Later, it was adapted to CIFAR-10 with **image size = 32x32, patch size = 4**.

#### **Training Details**
- **Optimizer**: AdamW (learning rate = 1e-3, weight decay = 0.05).
- **Loss Function**: Cross-Entropy Loss.
- **Scheduler**: Cosine Annealing with Warmup.
- **Batch Size**: 64.
- **Training Epochs**: 20.

---

### **Results and Observations**

MAE results

![](https://raw.githubusercontent.com/eracoding/rtml/main/code/a4/media/mae_results.png)

MAE loss graph

![](https://raw.githubusercontent.com/eracoding/rtml/main/code/a4/media/mae_pretrain.png)

MNIST accuracy graph

![](https://raw.githubusercontent.com/eracoding/rtml/main/code/a4/media/mnist_acc.png)

MAE loss graph

![](https://raw.githubusercontent.com/eracoding/rtml/main/code/a4/media/mnist_loss.png)

Cifar10 accuracy graph

![](https://raw.githubusercontent.com/eracoding/rtml/main/code/a4/media/cifar_acc.png)

Cifar10 loss graph

![](https://raw.githubusercontent.com/eracoding/rtml/main/code/a4/media/cifar_loss.png)

#### **MNIST Results**
| Epoch | Loss  | Train Accuracy | Val Accuracy |
|-------|-------|----------------|--------------|
| 1     | 0.4388 | 94.64%         | 98.92%       |
| 5     | 0.0316 | 99.13%         | 98.79%       |
| 10    | 0.0102 | 99.71%         | 99.10%       |
| 15    | 0.0013 | 99.97%         | 99.08%       |
| 20    | 0.0001 | 100.00%        | 99.22%       |

**Test Accuracy: 99.27%**

**Key Takeaways:**
- The model achieved near-perfect accuracy on MNIST.
- Minimal overfitting was observed due to the simplicity of the dataset.

---

#### **CIFAR-10 Results**
| Epoch | Loss  | Train Accuracy | Val Accuracy |
|-------|-------|----------------|--------------|
| 1     | 1.5792 | 42.04%         | 52.33%       |
| 5     | 0.9517 | 65.85%         | 64.48%       |
| 10    | 0.4399 | 84.60%         | 72.26%       |
| 15    | 0.0508 | 98.36%         | 73.50%       |
| 20    | 0.0003 | 100.00%        | 75.10%       |

**Test Accuracy: 75.12%**

**Key Observations:**
- Significant overfitting: Training accuracy reached **100%**, while validation accuracy stagnated at **~75%**.
- CIFAR-10 is more complex than MNIST, requiring a different training approach.
- Modifications such as changing patch size and initializing new positional embeddings led to a performance drop.

---

### Inference
Cifar-10 classification inference

![](https://raw.githubusercontent.com/eracoding/rtml/main/code/a4/media/inference.png)

### Conclusion

- MNIST Training: The ViT model successfully achieved high accuracy with minimal overfitting.
- CIFAR-10 Training: The model overfits due to dataset complexity and changes in patch representation.

### Final Thoughts
This experiment demonstrates how Vision Transformers excel on simpler datasets like MNIST but require careful tuning to generalize well to more complex datasets like CIFAR-10. Future improvements could push performance beyond 75%, making ViTs competitive for small-scale image classification tasks.
