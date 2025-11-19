---
layout: post
comments: true
title: Diffusion Models for Image Editing: A Study of SDEdit, Prompt-to-Prompt, and InstructPix2Pix
author: Ananya Sampat
date: 2025-11-18
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## 1. Introduction
Image editing stands at the heart of computer vision applications and enables object attribute modifications, style changes, or transformations in general appearance while retaining much of the structural information about an image. Classic deep learning methods, in particular the GAN-based approach, suffer from this balancing act. They either introduce artifacts, distort salient features, or fail to preserve the original content when performing edits.

Recently, diffusion models have emerged as a powerful alternative. Instead of generating an image with a single forward pass, diffusion models progressively remove noise according to a series of denoising steps. This iterative structure makes them particularly suitable for editing: partial noise levels can preserve the content, cross-attention layers control which regions change, and textual instructions guide the model to make targeted changes. For this reason, diffusion-based editing methods are among the most flexible and reliable tools in modern image manipulation.

In this report, I investigate three diffusion-based image editing methods-SDEdit, Prompt-to-Prompt, and InstructPix2Pix-each representing a different stage in the evolution of editing techniques. SDEdit showcases how diffusion is able to maintain structure during edits, Prompt-to-Prompt introduces fine-grained control through prompt manipulation, and InstructPix2Pix allows for natural-language-driven edits without large model retraining. All together, these works highlight the versatility of diffusion models and illustrate how iterative denoising can support a wide range of editing tasks.

## 2. Background 
Diffusion models are generative models that create images by learning to reverse a gradual noising process. During training, an image is repeatedly corrupted through the addition of Gaussian noise until it is almost indistinguishable from pure noise. Then the model is trained to do the opposite: starting from the noise, denoising it step by step to recover the distribution of original images. Although this idea is conceptually simple, the iterative nature of the denoising process makes diffusion models considerably more stable and controllable than earlier approaches to generative image synthesis.

### 2.1 Forward and Reverse Processes
The forward process progressively adds noise to an image over some predefined number of steps. Eventually, this results in a sample that bears no resemblance to the original content, but the noising schedule is carefully designed so the corruption is predictable. The reverse process, which the model learns, takes a noisy image and removes noise one step at a time. Each such denoising step is handled by a neural network, usually a U-Net, that predicts the noise added at that stage. The model can generate new images or modify existing ones by chaining these predictions together.

![Diffusion pipeline](/assets/images/teamXX/forward.png)
*Fig 1. Overview of the diffusion process. 
    Left: forward noising, where Gaussian noise is incrementally added to the image over $T$ steps. 
    Right: reverse denoising, which the model learns to perform step by step to reconstruct an image sample. 
    This formulation underlies all subsequent diffusion-based editing techniques. *

### Why Diffusion Models Support Image Editing

This is particularly effective for editing, since diffusion models allow fine control over how much of the original image to retain. By adjusting the amount of noise injected before running the reverse process, the model can either keep most of the structure intact-when the noise is low-or make more dramatic changes-when the noise is high. Further, many diffusion models make use of attention mechanisms, which naturally yield "handles" useful for directing edits. For instance, cross-attention maps link parts of a text prompt to specific regions of an image, allowing updates to be localized or constrained. Combining this preservation of structure with flexible conditioning allows diffusion models to perform a broad range of editing tasks. 
    
### Image Editing Task Categories

Editing images itself encompasses many types of transformations. Some edits aim to alter attributes-for example, color, lighting, or style-of an image, while the structure of the scene remains unchanged. Others include more semantic changes such as adding or removing objects or changing textures. More advanced methods support natural-language guidance where a user can describe the target edit in plain text. Diffusion-based methods have seen very successful applications across all these categories and often result in edits that preserve much more detail and coherence compared to earlier methods.

### Technical Perspective on Diffusion Models

Most models based on diffusion define the forward noising process as a sequence of Gaussian transitions:


\[
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I \right),
\]


Here, $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ denote the cumulative noise scheduling terms used in DDPMs.

 $\beta_t$ controls the amount of noise added at each step. The model then learns the reverse process $p_\theta(x_{t-1} \mid x_t)$, ussually parameterized by a U-Net that predicts the noise component $\epsilon_\theta(x_t, t)$. During either generation or editing, the model iteratively ``denoises'' a sample $x_T$ (pure noise) or a partially noised input image $x_t$ to reconstruct a coherent output.

This underlying formulation is common to the various editing methods discussed in later sections; each technique conditions or modifies this denoising process differently to achieve its specific editing behavior. 


Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

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
