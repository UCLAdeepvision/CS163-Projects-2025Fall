---
layout: post
comments: true
title: "Project Track: Project 8 - Streetview Semantic Segmentation"
author: "Team02"
date: 2025-12-12
---


> In this project, we delve into the topic of developing model to apply semantic segmentations on fine-grained urban structures based on pretrained Segformer model. We explore 3 approaches to enhance the model performance, and analyze the result of each.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Understanding street-level scenes through segmentation is crucial to autonoumous driving, urban mapping, and robot perception. And it is especially important when it comes to segment fine-grained urban structures because a lot of what makes a street safe, legal, and navigable lives in small, easily missed details. So, here is the core question we want to investigate in this project: How to improve the semantic segmentation performance on those fine-grained urban structures?

### Dataset
Cityscapes is a large-scale, widely used dataset for understanding complex urban environments, featuring diverse street scenes from many cities with high-quality pixel-level annotations for tasks like semantic segmentation and instance segmentation. It contains 30 classes and many of them are considered to be fine-grained urban structures, thus this dataset is a perfect choice for this project.

We remapped all categories in cityscapes to a consistent six-class scheme - fence, car, vegetation, pole, traffic sign, and traffic light. All of the classes are fine-grained urban structures. We define an additional implicit background class which are default to ignore by setting their pixel values to zero.

### Evaluation Metrics
We evaluate the performance of models using both per-class intersection over union (IoU) and mean intersection over union (mIoU). IoU measures the overlap between the predicted bounding box and the ground truth bounding box. The equation of calculating IoU is given as follows:

$$\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}$$


Where:
- $$A \cap B$$ is the area (or volume) of the overlap between **A** and **B**
- $$A \cup B$$ is the area (or volume) covered by **A** or **B** (their union)

Given **C** classes, the **mean IoU (mIoU)** is the average IoU across classes:

$$
\mathrm{mIoU} = \frac{1}{C}\sum_{c=1}^{C} \mathrm{IoU}_c
$$

The higher the IoU and mIoU value is, the better is the model performance.


## Model: Segformer

### Baseline

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

---
