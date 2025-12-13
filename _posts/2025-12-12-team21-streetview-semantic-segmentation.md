---
layout: post
comments: true
title: Street-View Semantic Segmentation
author: Andrea Asprer, Diana Chu, Matthew Lee, Tanisha Aggarwal
date: 2024-12-12
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
## Baseline CNN Model
## SAM/SAMV2 + Baseline CNN
This part of the project, the baseline CNN-based semantic segmentation model is extended by aplying the Segment Anything Model (SAM) and Segment Anything Model V2 (SAM V2) as post-processing refinement modules. All implementations and experiments described in this section are directly based on the code written and executed in the Jupyter Notebook (`.ipynb' file). The purpose of this integration is to evaluate whether using large, pretrained segmentation foundation models can improve the quality of the segmentation masks produced by a conventional CNN. 

The baseline model used in the notebook is a pretrained Fully Convolutional Network (FCN) with a ResNet-50 backbone. This model is loaded from standard deep learning libraries and applied directly to the dataset to generate initial semantic segmentation masks. The typical CNN-based segmentation model uses an encoder that repeatedly downsampled the image and the decoder then upsample these rough features into a dense segmentation mask where each pixel gets a class score. However, downsampling causes thin structures like poles or fences disappear in low-res feature maps. Visual inspection in the notebook shows that this in turn results in blurring or blending the small objects to its surrounding pixels. 

The Segment Anything Model (SAM) is then applied on top of the baseline CNN predictions. In the notebook, SAM is loaded using the official implementation and pretrained weights provided by the SAM GitHub repository, which is linked directly in the code. The corresponding model configuration files are also taken from the same repository to ensure correct model initialization. The baseline CNN output is used to guide SAM by defining regions of interest, and SAM generates refined segmentation masks for these regions. This process allows SAM to leverage its strong generalization ability while still being guided by the CNN's semantic predictions. 

Segment Anything Model V2 (SAM V2) is applied in a similar manner in the notebook. The SAM V2 model and its configuation files are obtained from the official SAM V2 GitHub repository, which is also referenced in the notebook. Compared to the original SAM, SAM V2 provides more stable mask predictions and improved handling of complex scenes. When applied to the CNN-generated masks, SAM V2 produces cleaner segmentation outputs with better-defined object boundaries and reduced background noise. 

To quantitatively evaluate whether SAM and SAM V2 improve segmentation performance, the mean intersection over union (mean IoU) metric is used in the notebook. Mean IoU measures the overlap between the predicted segmentation masks and the ground truth masks across all classes. For each class, IoU is computed as the ratio between the intersection and the union of the predicted and the ground truth regions, and the final mean IoU is obtained by averaging over all classes. By comparing the mean IoU values of the baseline CNN, CNN + SAM, and CNN + SAM V2, the notebook evaluates whether the refined masks produced by SAM-based models lead to measurable improvements. 

The experimental results show that applying SAM and SAM V2 on top of the baseline CNN leads to higher mean IoU scores, indicating better alignment between predicted masks and ground truth. Among the tested approaches, the CNN combined with SAM V2 achieves the best overall performance, both qualitatively and quantitatively. These results suggest that SAM-based refinement is an effective way to enhance CNN-based semantic segmentation without retraining the original model. 

## Apply CRF and Grounded SAM on Baseline CNN
## Conclusion
## Reference
## Demo Video
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
