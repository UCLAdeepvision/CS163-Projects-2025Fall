---
layout: post
comments: true
title: Post Template
author: Radhika Kakkar, Janie Kuang, Nyla Zia
date: 2024-12-13
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Motivation
Medical segmentation plays a key role in decision making in the clinical setting. It supports tasks ranging from diagnosis, treatment planning, and disease monitoring. However, the existence of large variability in imaging modalities, anatomical structures, and labeling conventions makes segmentation really difficult. Current existing approaches are catered towards narrowly defined tasks that still require a large amount of labelled data and retraining for new tasks. Furthermore, new segmentation tasks come to light frequently that differ in both appearance and semantic definition from previous training data. These tasks may be centered on uncommon autonomies, institution specific labelling protocols, or generally unseen imaging conditions. This leads to overly relying on very detailed task specific annotations or retraining continuously.  This all makes it very difficult to scale medical image segmentation across diverse, real world clinical settings.

These challenges motivate the need for segmentation frameworks that are able to generalize well across new tasks in an efficient and effective manner. Ideally, these systems would support different formulations of task specification while simultaneously maintaining good performance across heterogeneous medical imaging domains. The following sections deep dive into three approaches that attempt to address these challenges from different yet complimentary perspectives. 

## MedSAM
Current medical segmentation models lack generalizability as many are tailored to individual medical image segmentation tasks. This can prove to be an obstacle for practical use of models in a clinical setting when there is a wide range of applications where medical segmentation can be applied. There already exist generalized segmentation models like the segment anything model (SAM) which is a general segmentation foundation model that works on medical image segmentations but performs conditionally with weaker performance when targets are characterized by weak boundaries. This led to the development of MedSAM, a promptable segmentation model that can segment medical images both 2D and 3D(processes 3D images as a series of 2D slices) with the enhanced aid of specificity through user prompting.

![YOLO]({{ '/assets/images/team45/Figure1.png' | relative_url }})
{: style="width: 600px; max-width: 100%; margin: 0 auto; display: block;"}

<p style="text-align: center;"><em>Fig 1. a. MedSAM dataset distribution 
b. MedSAM model architecture</em> [1].</p>

MedSAM’s approach was to finetune SAM on more than one million medical image mask pairs. To start, 1.5 million medical image mask pairs were split for training, tuning, and validation of the model. These medical images range from a variety of medical imaging modalities including CT, MRI, X-ray, ultrasound, endoscopy, ultrasound, mammography, OCT, and pathology and also cover a diverse range of targets like organs, tissues, and lesions to ensure generalization. The data is then trained on a model retaining most of SAM’s backbone which includes a ViT-Base model with an image encoder that extracts dense visual features, a prompt encoder that processes user inputs, and a mask decoder that generates segmentation outputs. During training, bounding box prompts also help the model learn mapping by deriving ground truth masks which help convert coarse spatial localization to precise boundary delineation. This is aimed to help increase human interaction as in many medical settings, different clinicians may have different needs from the same image.

![YOLO]({{ '/assets/images/team45/Figure2.png' | relative_url }})
{: style="width: 600px; max-width: 100%; margin: 0 auto; display: block;"}

<p style="text-align: center;"><em>Fig 2. Performance Correspondence of Internal Validation Tasks across the models </em> [1].</p>

![YOLO]({{ '/assets/images/team45/Figure3.png' | relative_url }})
{: style="width: 600px; max-width: 100%; margin: 0 auto; display: block;"}

<p style="text-align: center;"><em>Fig 3. Performance Correspondence of External Validation Tasks across the models </em> [1].</p>

To evaluate the effectiveness of MedSAM, its performance was compared against SAM and specialist models like U-Net and DeepLabV3+. Each of the specialist models were trained separately for each of the modalities that MedSAM is trained on and then tested with internal and external validation where internal validation was the default data splits from the same dataset and the external validation came from unseen, new datasets of the same modality. Results then showed that MedSAM had a stronger consistent performance on internal and validation while U-Net and DeepLabV3+ would perform stronger on internal validation but poorly on external validation showing their limited generalization ability. SAM performed relatively poorly across all validation sets and in comparison to the other three models. 

In general, MedSAM is a strong step towards making segmentation foundation models useful in medicine and it shows that with large, well-curated medical data, a generalized segmentation model in medical uses is possible and broadly useful. 

## UniverSeg
The power of large scale foundational models for medical image segmentation is clearly demonstrated through MedSAM. However, the reality is that it still relies on predefined prompt formats to adapt to new segmentation issues and requires task specific fine tuning. UniverSeg tackles the same ultimate goal of universality from a different and interesting angle. Instead of prompt engineering or retraining, it proposes an adaptation framework that is trained only once and is inference only. It essentially enables the segmentation of new medical tasks without additional learning. 

![YOLO]({{ '/assets/images/team45/Figure4.png' | relative_url }})
{: style="width: 600px; max-width: 100%; margin: 0 auto; display: block;"}

<p style="text-align: center;"><em>Fig 4. Train and Test Segmentation Tasks </em> [1].</p>

UniverSeg approaches medical image segmentation as a task conditioned inference problem. It does not encode the task implicitly using the weights of the model. Instead, the task is defined explicitly at inference time using a support set, which is a small collection of labelled image-mask pairs that represent the segmentation objective at hand. Given this support set and a query image, the model is able to output a segmentation for the query in a single forward pass. Essentially, if a few representative examples are given at inference time, UniverSeg is able to generalize to unseen anatomies, image modalities, and label spaces. 

From an architectural perspective, UniverSeg contains a symmetric encoder that is responsible for processing both query and support images. This is then followed by a cross-feature interaction module, which is referred to as CrossConv or CrossBlock. This module is responsible for the facilitation of the bidirectional exchange of information between the query and support features. This essentially allows the model to understand what structures should be segmented in the query image based on the support examples available. Most importantly, this model does not assume that there are a fixed number of classes or predefined semantics. This makes it very well suited for heterogeneous and open set medical segmentation tasks. 

![YOLO]({{ '/assets/images/team45/Figure5.png' | relative_url }})
{: style="width: 600px; max-width: 100%; margin: 0 auto; display: block;"}

<p style="text-align: center;"><em>Fig 5. UniverSeg network (left) and CrossBlock architecture (right) </em> [1].</p>

The authors curated a very large and diverse training corpus named MegaMedical in order to train a model capable of broad generalization. MegaMedical consists of 53 public medical segmentation datasets. It spans various anatomical targets and imaging modalities (MRI, CT, fundus imaging, etc.). During the training process, segmentation tasks are sampled episodically. Each of the episodes define a new task through a randomly selected support set and the corresponding query images. This set up was designed to mirror the setting during inference. It really encourages the model to learn segmentation principles that are task agonistic instead of memorizing data specific patterns. 

From an empirical standpoint, UniverSeg showcases both strong zero shot and few shot performance when tested on a large range of previously unseen data sets. The model was evaluated on segmentation tasks that differed greatly from the training distribution in terms of label definitions, autonomy, and modality. Furthermore, the results clearly show that UniverSeg outperforms traditional supervised baselines that have been trained on limited data. It is also competitive with or superior to several of the specialized few shot medical segmentation methods. It is crucial to note this performance is achieved without any fine tuning, truly highlighting the effectiveness of task conditioning through support examples. 

![YOLO]({{ '/assets/images/team45/Table1.png' | relative_url }})
{: style="width: 500px; max-width: 100%; margin: 0 auto; display: block;"}

<p style="text-align: center;"><em>Table 1: Performance Summary of UniverSeg and each FS baseline (right) </em> [1].</p>

![YOLO]({{ '/assets/images/team45/Figure6.png' | relative_url }})
{: style="width: 600px; max-width: 100%; margin: 0 auto; display: block;"}

<p style="text-align: center;"><em>Fig 6. Average Dice score per each held out dataset </em> [1].</p>

While MedSAM focuses on promptable segmentation within a powerful model that has been pretrained, UniverSeg transfers the burden of adaptation from the parameters of the model to contextual examples. This represents an important design tradeoff in the realm of universal medical segmentation. MedSAM performs extremely well when simple prompts and large pretrained representations prove to be more than sufficient. On the other hand, UniverSeg truly shines in cases where the new tasks can not be encoded through prompts easily, labeled examples are present, and retraining is ineffective or inefficient. Together, these two approaches intertwine and provide interesting insights into complementary future work focusing on scalable and flexible medical image segmentation. 


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
