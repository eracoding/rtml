# Self-Supervised Learning Methods: Summary

### Momentum Contrast (MoCo) - [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722)
#### Main ideas, techniques, and approaches:
1. MoCo main algorithm - build dynamic dictionary with queue and moving-averaged encoder to enable large and consistent dictionary on-the-fly (queue + momentum encoder - moving-averaged) that facilitates contrastive unsupervised learning
2. Outperform its supervised pre-training counterpart in 7 detection/segmentation tasks on PASCAL VOC, COCO
3. Importance of unsupervised learning in computer vision - Language tasks have discrete signals spaces (words, sub-words, units, etc.) for building tokenized dictionaries, on which unsupervised learning can be based, but in cv, raw signal is in a continuous, high-dimensional space and is not structured for human communication.
4. Unsupervised learning trains encoder to perform dictionary look-up - encoded query should be similar to its matching key and dissimilar to others (contrastive learning). Learning is formulated as minimizing a contrastive loss.
5. Main purpose of unsupervised learning is to pre-train representations (features). MoCo pre-trained on ImageNet (1.3 million data, 1k classes) and Instagram (1 billion images) datas. MoCo largely close the gap between unsupervised and supervised representation learning in many computer vision tasks.
6. Unsupervised (self-supervised) learning methods generally involve two aspects - pretext tasks and loss function. **Pretext Tasks* implies that the task is being solved is not of genuine interest, but is solved only for the true purpose of learning a good data representation. MoCo focuses on the loss function aspect.
7. Good thoughts regarding loss function fundamental meaning - common way of defining a loss function is to measure the difference between a model's prediction and a fixed target, such as reconstructing the input pixels (auto-encoder) by L1 or L2 losses, or classifying the input into pre-defined categories by cross entropy or margin-based losses.
**Contrastive Losses measure the similarities of sample pairs in a representation space.** Instead of matching input to a fixed target, in contrastive loss formulations the target can vary on-the-fly during training and can be defined in terms of the data representation computed by network. **Adversarial losses measure the difference between probability distributions.**
8. Noise-Contrastive Estimation (NCE)
9. Using general pretext task - instance discrimination method + NCE.
10. Main idea behind contrastive learning: "encoded query q and a set of encoded samples that the keys of dictionary. Assume that there is a single key (denoted as $k_+$) in the dictionary that q matches. Contrastive loss is a function whose value is low when q is similar to its positive key $k_+$ and dissimilar to all other keys (negative keys for q). With similarity by dot product, a form of contrastive loss function (InfoNCE) is considered:" $$L_q = -\log \dfrac{\exp(q\cdot k_+/ \tau)}{\sum_{i=0}^K \exp(q\cdot k_i / \tau)}$$ 
where $\tau$ is temperature hyper-parameter per.
11. Main hypothesis - good features can be learned by a large dictionary that covers a rich set of negative samples, while the encoder for the dictionary keys is kept as consistent as possible despite its evolution.
12. Ablation study - systematically remove or modify compoents of model or system to analyze their impact on performance (understand which parts contribute most)
13. It used ResNet as encoder with changing last fc layer with fixed-dimensional output (d=128) which is normalized by L2-norm. Temperature=0.07, data augmentation - crop randomly 224x224, shuffle batch normalization to deal data leakage (or as they called "cheat" behavior of pretext task).

---

### **SimCLR - [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709)**
#### **Main Ideas, Techniques, and Approaches**
1. **Contrastive Learning with Augmentations**  
   - SimCLR learns representations by maximizing agreement between differently augmented views of the same image.
   - Uses **strong data augmentation**, including random crop, color distortion, and Gaussian blur.

2. **Framework Structure**  
   - Composed of a **base encoder** (ResNet) and a **nonlinear projection head** (MLP with ReLU) that maps the representation to a contrastive loss space.
   - The projection head helps improve performance, and representations are taken from the encoder before the projection layer.

3. **Contrastive Loss - InfoNCE**  
   - Given a batch of images, two views per image are generated via augmentation.
   - Each pair is treated as a positive sample, and the rest of the batch forms negative samples.
   - The **InfoNCE loss** is used to maximize similarity between positive pairs and minimize similarity with negative pairs:
     $$
     L_q = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i=0}^K \exp(q \cdot k_i / \tau)}
     $$
     where \( \tau \) is a temperature hyperparameter.

4. **Key Contributions**  
   - Demonstrates that **large batch sizes** (4096 or more) help in contrastive learning.
   - Shows that **strong data augmentation** is crucial for contrastive learning success.
   - Projects features into a higher dimensional space for contrastive loss but discards the projection head during downstream tasks.

---

### **BYOL - [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/pdf/2006.07733)**
#### **Main Ideas, Techniques, and Approaches**
1. **Contrastive Learning Without Negative Samples**  
   - Unlike SimCLR and MoCo, BYOL does **not** use negative samples.
   - Learns representations by **predicting one view of an image from another view**.

2. **Framework Structure**  
   - Two networks:  
     - **Online Network** (encoder + predictor)  
     - **Target Network** (encoder, no predictor, updated with momentum)  
   - Online network learns to predict the output of the target network.

3. **Loss Function**  
   - L2 loss between the online network's projection and the target network's output:
     $$
     L = || q_{\theta} - z_{\xi} ||^2
     $$
     where:
     - \( q_{\theta} \) is the online network's output
     - \( z_{\xi} \) is the target network's output

4. **Momentum Update**  
   - The target network parameters \( \xi \) are **updated slowly** using an exponential moving average of the online network parameters \( \theta \):
     $$
     \xi \leftarrow m \xi + (1 - m) \theta
     $$
   - This prevents collapse and improves stability.

5. **Key Contributions**  
   - Removes the need for **negative pairs** in contrastive learning.
   - Demonstrates that **predicting representations** can prevent representation collapse.
   - Outperforms MoCo and SimCLR while using smaller batch sizes.

---


### **Barlow Twins - [Reducing the Dimensional Collapse in Self-Supervised Learning](https://arxiv.org/pdf/2103.03230)**
#### **Main Ideas, Techniques, and Approaches**
1. **Key Motivation**  
   - Prevents **dimensional collapse**, where self-supervised methods collapse all features to similar values.

2. **Framework Structure**  
   - Two identical networks process different augmentations of the same image.
   - The outputs are projected into an embedding space and compared using a **cross-correlation matrix**.

3. **Loss Function: Barlow Twins Loss**  
   - Encourages **high similarity** for positive pairs and **decorrelates different features**:
     $$
     L = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i \neq j} C_{ij}^2
     $$
     where:
     - \( C \) is the cross-correlation matrix between embeddings of the two views.
     - The first term ensures diagonal elements are close to 1 (high similarity).
     - The second term forces off-diagonal elements to be close to 0 (decorrelation).

4. **Key Contributions**  
   - Eliminates the need for negative samples.
   - Ensures that learned representations **capture all feature dimensions**.
   - Matches or outperforms SimCLR and BYOL on downstream tasks.

---


### **SwAV - [Swapping Assignments between Views for Learning Representations](https://arxiv.org/pdf/2006.09882)**
#### **Main Ideas, Techniques, and Approaches**
1. **Cluster-Based Self-Supervised Learning**  
   - Uses a clustering-based approach instead of direct instance discrimination.
   - Each image representation is assigned to a cluster **prototype**.

2. **Framework Structure**  
   - Uses a **swapped prediction mechanism** where two augmented views of the same image are assigned **pseudo-labels (cluster assignments)**.
   - The model learns to **predict the pseudo-label** of one view from the other.

3. **Loss Function: Swapped Prediction Loss**  
   - Instead of directly comparing embeddings, SwAV **predicts cluster assignments**.
   - Uses **online clustering** and **Sinkhorn-Knopp optimization** to compute soft assignments.

4. **Key Contributions**  
   - Avoids large batch sizes by leveraging clustering techniques.
   - Works well with **multi-crop augmentations**, where smaller crops help learn local features.
   - Achieves high performance with smaller computational cost compared to contrastive methods.

---

---

## **Summary of Differences**
| Method       | Uses Negative Pairs? | Loss Function | Main Innovation | Published Date |
|-------------|-----------------|--------------|----------------|---------|
| **MoCo**    | ✅ Yes | Contrastive (InfoNCE) | Momentum-based dictionary to maintain large negative sample pool | 23 Mar 2020 |
| **SimCLR**  | ✅ Yes | Contrastive (InfoNCE) | Strong augmentations, large batch sizes | 1 July 2020 |
| **BYOL**    | ❌ No  | L2 Loss | Self-distillation with momentum encoder (no negatives) | 10 Sep 2020 |
| **Barlow Twins** | ❌ No  | Cross-Correlation Loss | Prevents collapse by decorrelating features | 8 Jan 2021 |
| **SwAV**    | ❌ No  | Swapped Assignment Loss | Clustering-based self-supervised learning | 14 Jun 2021 |

---
