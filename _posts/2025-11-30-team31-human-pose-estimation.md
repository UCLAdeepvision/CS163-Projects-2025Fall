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

Accompanying the implementation is a curated collection of seminal and research papers, datasets, benchmark tasks, and open-source implementations that cover 2D and 3D pose estimation, human mesh construction, pose-based action recognition, and video pose tracking. These sources provide researchers and engineers with a structured overview of the theoretical foundations, methodological advances, and practical tools in the domain. In combiningthe  implementation toolkit with a comprehensive research source, one obtains both the practical means to build pose-estimation systems and the theoretical grounding to understand trade-offs. In this paper, we leverage the idea that we adopt the MMPose framwork for our pose estimation tasks, while consulting the literature summarized by several resources to choose appropraite architectures, training strategies, and evaluation protocols. The goal is to demonstrate accurate pose detection in both 2D and 3D, under diverse conditions, and to assess how well modern models generalize beyond standard benchmark datasets. 

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
