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

Accompanying the implementation is a curated collection of seminal and research papers, datasets, benchmark tasks, and open-source implementations that cover 2D and 3D pose estimation, human mesh construction, pose-based action recognition, and video pose tracking. These sources provide researchers and engineers with a structured overview of the theoretical foundations, methodological advances, and practical tools in the domain. In combiningthe  implementation toolkit with a comprehensive research source, one obtains both the useful means to build pose-estimation systems and the theoretical grounding to understand trade-offs. In this paper, we leverage the idea that we adopt the MMPose framwork for our pose estimation tasks, while consulting the literature summarized by several resources to choose appropraite architectures, training strategies, and evaluation protocols. The goal is to demonstrate accurate pose detection in both 2D and 3D, under diverse conditions, and to assess how well modern models generalize beyond standard benchmark datasets. 

## Why Does Human Pose Estimation Work Now? 
Although pose estimation has been studied for decades, its recent success can largely be attributed to three converging facts. These factors are data, computation, and model design. Modern pose estimation models rely on large-scale annotated datasets such as COCO, MPII, Human3.6M, and 3DPW, which provide diverse human poses across different viewpoints, environments, and levels of occlusion. Without these datasets, learning representations of human articulation would not be possible. 

Just as important is the availability of powerful computational resources. Training deep neural networks for pose estimation involves optimizing millions of parameters and processing high-resolution feature maps. GPUs and specialized accelerators make it possible to train models efficiently and deploy them in real-time systems. 

Finally, deep learning architectures are designed to learning spatial dependencies between joints. Unlike traditional hand-crafted approaches, deep models can automatically learn hierarchical representations that encode both local joint appearance and global body structure. This allows pose estimation systems to scale effectively as data size increases, improving performance rather than saturating.  

## Problem Formulation of Human Pose Estimation 
At its core, human pose estimation can be formulated as a structured prediction problem. Given an input Image $$I$$, the goal is to predict a set of keypoints: 

$$
\mathbf{P} = P = \{(x_i, y_i, c_i)\}_{i=1}^{K}
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

where $$f_theta$$ is a CNN parameterized by $$theta$$. The loss function is usually defined as the mean squared error between predicted and ground-truth heatmaps: 

$$
\mathcal{L} =
\frac{1}{K} \sum_{k=1}^{K}
\left\| \hat{\mathbf{H}}_k - \mathbf{H}_k \right\|_2^2
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

where $$g_phi$$ is a regression network. The loss is typically defined as the mean per-joint position error (MPJPE): 

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
