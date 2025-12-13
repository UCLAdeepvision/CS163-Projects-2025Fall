---
layout: post
comments: true
title: From Paris to Seychelles - Deep Learning Techniques for Global Image Geolocation
author: Shelby Falde, Joshua Li, Alexander Chen
date: 2025-12-12
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

Image geolocation is the task of predicting the geographic coordinates (latitude and longitude) of a given image. While humans rely on specific landmarks or cultural context, training a machine to recognize any location on Earth is a massive classification and regression challenge.

Early approaches relied on matching images to databases of landmarks, but these failed in "in the wild" scenarios where no distinct landmark is visible (e.g., a random road in rural Norway). Deep learning has since shifted this paradigm by learning global feature distributions and characteristics around the world.

In this survey, we explore how architectures have evolved from Convolutional Neural Networks (CNNs) to Transformers and CLIP-based models, specifically focusing on how they handle the partitioning of the Earth and the loss functions used to train them.

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
