---
layout: post
comments: true
title: Human Pose Estimation
author: Lina Lee   
date: 2025-12-12
---


> In this paper, I will be discussing the fundamentals and workings of deep learning for human pose estimation. I believe that there has been a lot of research and breakthroughs, especially recently, on technology that relates to this, and I hope that this deep dive will bring some clarity and new information to how it works!

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Human pose estimation is the computational task of detecting and localizing predefined keypoints, such as joints or landmarks, on one or more objects in images or videos. These keypoints are typically represented as 2D or 3D coordinates (e.g., [x, y] or [x, y, visibility]), often accompanied by confidence scores indicating the model’s certainty for each point. By capturing the spatial arrangement of the body parts, pose estimation enables fine-grained applications such as motion analysis, gesture recognition, animation, biomechanics, surveillance, and human-computer interaction.   

The field of pose estimation has grown tremendously in the past few years, driven by advances in deep learning, the availability of large annotated datasets, and the development of flexible, production toolkits. Out of all of these, MMPose stands out as an open-source and extensible framework build on PyTorch that supports a wide array of tasks. Some of these tasks include 2D multi-person human pose estimation, hand keypoint detection, face landmarks, full-body pose estimation including body, hands, face, and feet, animal pose estimations, and so much more. 

The advantages of MMPose is its comprehensive "model zoo" that includes both accuracy-oriented and real-time lightweight architectures, pertained weights on strandard datasets, and configurable pipelines for dataset loading, data augmentations, and evaluation. This versatility makes MMPose suitable for both academic research and real-world production systems, whether the task is single-person pose detection, multi-person tracking, or whole-body landmark detection. 

![YOLO]({{ '/assets/images/assets/images/905972224/map.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: UMAP of datasets with root subtraction* [1].


Accompanying the implementation is a curated collection of seminal and research papers, datasets, benchmark tasks, and open-source implementations that cover 2D and 3D pose estimation, human mesh construction, pose-based action recognition, and video pose tracking. These sources provide researchers and engineers with a structured overview of the theoretical foundations, methodological advances, and practical tools in the domain. In combiningthe  implementation toolkit with a comprehensive research source, one obtains both the useful means to build pose-estimation systems and the theoretical grounding to understand trade-offs. In this paper, we leverage the idea that we adopt the MMPose framwork for our pose estimation tasks, while consulting the literature summarized by several resources to choose appropraite architectures, training strategies, and evaluation protocols. The goal is to demonstrate accurate pose detection in both 2D and 3D, under diverse conditions, and to assess how well modern models generalize beyond standard benchmark datasets. 

## Why Does Human Pose Estimation Work Now? 
Although pose estimation has been studied for decades, its recent success can largely be attributed to three converging facts. These factors are data, computation, and model design. Modern pose estimation models rely on large-scale annotated datasets such as COCO, MPII, Human3.6M, and 3DPW, which provide diverse human poses across different viewpoints, environments, and levels of occlusion. Without these datasets, learning representations of human articulation would not be possible. 

Just as important is the availability of powerful computational resources. Training deep neural networks for pose estimation involves optimizing millions of parameters and processing high-resolution feature maps. GPUs and specialized accelerators make it possible to train models efficiently and deploy them in real-time systems. 

Finally, deep learning architectures are designed to learning spatial dependencies between joints. Unlike traditional hand-crafted approaches, deep models can automatically learn hierarchical representations that encode both local joint appearance and global body structure. This allows pose estimation systems to scale effectively as data size increases, improving performance rather than saturating.  

## Problem Formulation of Human Pose Estimation 
At its core, human pose estimation can be formulated as a structured prediction problem. Given an input Image $$I$$, the goal is to predict a set of keypoints: 

$$
\mathbf{P} = \{(x_i, y_i, c_i)\}_{i=1}^{K}
])
$$

where $$K$$ is the number of keypoints, $$x_i, y_i)$$ denotes the spatial location of the $$i$$ -th joint, and $$c_i$$ represents either a confidence score or a visibility flag. 

Most modern approaches model pose estimation as a heatmap regression problem. For each joint $$i$$, the network predicts a heat map (shown  below), where each pixel value represents the probability of that joint appearing at that location. The final keypoint location is obtained by: 

$$
(x_i, y_i) = \arg\max_{(x, y)} H_i(x, y), \quad H_i \in \mathbb{R}^{H \times W}
$$

This formulation is particularly effective because it preserves spatial uncertainty and allows the network to express ambiguity when joints are occluded or visually similar. 

### Single-Person and Multi-Person Pose Estimation 
Single-person pose estimation assumes the presence of one dominant subject in the image. The model focuses entirely on accurately localizing all keypoints of that individual, often after a preprocessing step that crops the person from the background. This setup allows for high precision and is commonly used in controlled environments such as motion capturing or sports analysis. 

Multi-person pose estimation, on the other hand, introduces the challenge of associating detected keypoints with the correct individuals. Top-down approaches first detect bounding boxes for each person and then apply a single-person pose estimator to each crop. Bottom-up approaches detect all keypoints in the image simultaneously and then group them into individual skeletons. 

The top-down methods generally achieve higher accuracy but scale linearly with the number of people. Bottom-up methods are more computationally efficient for crowded scenes but require robust association algorithms. 

### Deep Learning Models for Pose Estimation
Convolutional neural networks form the backbone of most pose estimation systems. CNNs exploit local spatial correlations in images and progressively build higher-level features through stacked convolutional layers. Early layers capture low-level patterns such as edges, while deeper layers encode semantic concepts like limbs and body parts.  

A typical pose estimation network predicts a stack of heatmaps using a fully convolutional architecture: 

$$
\hat{\mathbf{H}}_k = f_\theta(\mathbf{I})
$$

where $$f_\theta$$ is a CNN parameterized by $$theta$$. The loss function is usually defined as the mean squared error between predicted and ground-truth heatmaps: 

$$
\text{L} = \frac{1}{K} \sum_{i=1}^{K} \left\| H_i - \hat{H}_i \right\|_2^2
$$

## Transformer-Based Pose Models 
Transformers introduce self-attention mechanisms that allow each join representation to attend to all others. This is particularly useful for modeling long-range dependencies, such as symmetric limbs or full-body constraints. Instead of relying solely on local convolutions, transformers explicitly reason about global body structure. 

Self-attention is written as: 

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

where $$Q$$, $$K$$, and $$V$$, are query, key, and value matrices derived from feature embeddings. By inforporating attention, pose models can better handle occlusions and complex interactions between joints. 

## 3D Human Pose Estimation 
3D pose estimation extends the 2D formulation by predicting depth information for each joint. The task is inherently ill-posed because multiple 3D poses can project to the same 2D configuration. Many approaches therefore, rely on multi-view supervision, temporal constraints, or learned priors over human anatomy. 

A common formulation predicts 3D joints ($$x_i$$, $$y_i$$, $$z_i$$) from 2D detections: 

$$
P_\text{3D} = g_\phi(P_\text{2D})
$$

where $$g_\phi$$ is a regression network. The loss is typically defined as the mean per-joint position error (MPJPE): 

$$
\text{MPJPE} = \frac{1}{K} \sum_{i=1}^{K} \| \hat{p}^i - p^i \|_2
$$

## Implementation Example with MMPose 
Below is a simplified example of running inference using a pretrained MMPose model: 
```
from mmpose.apis import init_pose_model, inference_top_down_pose_model
from mmdet.apis import init_detector

detector = init_detector(det_config, det_checkpoint)
pose_model = init_pose_model(pose_config, pose_checkpoint)

person_results = inference_detector(detector, image)
pose_results, _ = inference_top_down_pose_model(
    pose_model,
    image,
    person_results
)
```
This pipeline demonstrates how detection and pose estimation are decoupled in a top-down framework, enabling flexible experimentation and modular design. 

## Applications of Human Pose Estimation 
Human pose estimation enables a wide range of applications beyond academic benchmarks. In healthcare, pose estimation supports rehabilitation monitoring and gait analysis. In sports, it enables fine-grained motion analysis for performance optimization. In entertainment and AR/VR, pose estimation allows realistic avatar animation and immersive interaction.

By converting raw pixel data into structured skeletal representations, pose estimation serves as a bridge between perception and semantic understanding of human behavior.

## Related Work 
Human pose estimation has evolved considerably over the last decade, largely driven by advances in deep learning architectures. In this section, we compare some of the most influential models and frameworks, including SRCNN and ViTPose, highlighting their design and limitations. 

### SRCNN (Super-Resolution Convolutional Neural Network) 
Although SRCNN was originally designed for image super-resolution, its underlying convolutional architecture has inspired early approaches to pose estimation through feature extraction and heatmap regression. SRCNN consists of three layers of convolution:

$$
F_1(Y) = \max\left(0, W_1 * Y + B_1\right)
$$

$$
F_2(Y) = \max\left(0, W_2 * F_1(Y) + B_2\right)
$$

$$
F(Y) = W_3 * F_2(Y) + B_3
$$

Where $$Y$$ is the input image, $$*$$ denotes convolution, $$W_i$$ and $$B_i$$ are the weights and biases of the $$i$$-th layer, and max (0,⋅) represents the ReLU activation. SRCNN's simplicity allows fast training and easy integration into pipelines where low-resolution keypoint heatmaps are upscaled to higher resolution for finer localization. 

### Implementation of SRCNN 
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
```
While SRCNN provides a foundation for feature extraction, it suffers in hgih complexity scenes due to its shallow structure and limited capacity to capture long range dependencies. 

### ViTPose (Vision Transformer for Pose Estimation)
ViTPose leverages transformer architectures, which excel at modeling long-range dependencies and global context, complementing convolutional methods. The input image is split into patches, embedded, and passed through a series of self-attention layers:

$$
Z_0 = X_p + E_{\text{pos}}
$$

$$
Z_l' = \text{MSA}(\text{LN}(Z_{l-1})) + Z_{l-1}
$$

$$
Z_l = \text{MLP}(\text{LN}(Z_l')) + Z_l'
$$

Where $$X_p$$ is the patch embedding, $$E_{\text{pos}}$$ is positional encoding, MSA is multi-head self-attention, and LN is layer normalization. ViTPose produces heatmaps for keypoints using transformer output, enabling it to capture spatial and contextual dependencies across the whole image, even in crowded scenes. 
```
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x
```
Example of patch embedding layer: 
```
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x
```
Self-attention allows every patch to attend to every other patch: 
```
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out
```
ViTPose has demonstrated strong performance on multi-person and whole-body pose estimation, particularly in complex scenes where CNNs struggle to capture long-range interactions. In modern pose estimation pipelines, a common strategy is to combine convolutional backbones with transformer modules, achieving both fine-grained localization and robust global reasoning.

## Conclusion 
Human pose estimation has emerged as a fundamental problem in modern computer vision, driven by deep learning, large datasets, and powerful toolkits such as MMPose. Through structured representations of the human body, pose estimation enables machines to reason about motion, posture, and interaction at a fine-grained level. As models continue to evolve toward 3D, whole-body, and real-time systems, pose estimation will remain a critical component of intelligent visual understanding. 




## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

---
