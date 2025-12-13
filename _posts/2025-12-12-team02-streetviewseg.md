---
layout: post
comments: true
title: "Project Track: Project 8 - Street-view Semantic Segmentation"
author: "Team02"
date: 2025-12-12
---


> In this project, we delve into the topic of developing model to apply semantic segmentations on fine-grained urban structures based on pretrained Segformer model. We explore 3 approaches to enhance the model performance, and analyze the result of each.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Understanding street-level scenes through segmentation is crucial to autonoumous driving, urban mapping, and robot perception. And it is especially important when it comes to segment fine-grained urban structures because a lot of what makes a street safe, legal, and navigable lives in small, easily missed details. So, this naturally leads to the question we want to investigate in this project: How to improve the semantic segmentation performance on those fine-grained urban structures?

## Dataset
Cityscapes is a large-scale, widely used dataset for understanding complex urban environments, featuring diverse street scenes from many cities with high-quality pixel-level annotations for tasks like semantic segmentation and instance segmentation. It contains 30 classes and many of them are considered to be fine-grained urban structures, thus this dataset is a perfect choice for this project.

We remapped all categories in cityscapes to a consistent six-class scheme - fence, car, vegetation, pole, traffic sign, and traffic light. All of the classes are fine-grained urban structures. We define an additional implicit background class which are default to ignore by setting their pixel values to zero.
### Dataset Split
We partition the Cityscapes dataset into three subsets from training, validation, and testing. From the original Cityscapes training split, we sample 2000 images to form our training set, 250 images to form our validation set, and another 250 images to form our test set.

## Model: Segformer
In this project, we build everything upon the Segformer model.
Segformer is a transformer-based semantic segmentation model designed to be simple and accurate. It contains two main parts: encoder and decoder. 
The encoder is MiT (Mix Transformer), a hierarchical Transformer that produces 4 multi-scale feature maps. It uses overlapped patch embeddings and an efficient attention design.
The decoder is a lightweight All-MLP decoder. It linearly projects each of the 4 feature maps to the same channel size, upsamples them to the same resolution, concatenates and fuses them with an MLP, then outputs per-pixel class scores
![Segformer]({{ '/assets/images/team02/segformerArch.png' | relative_url }}){: style="width: 400px; max-width: 100%;"}
*Fig 1. Segformer Architecture, consists of a hierarchical Transformer encoder to extract coarse and fine features and a lightweight All-MLP decoder to fuse these multi-level features and predict the segmentation mask* [1].

## Evaluation Metrics
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

## Baseline Methods
We use a fully fine-tuned SegFormer-B0 segmentation model from a Cityscapes-fine-tuned checkpoint, with a newly initialized 7-class (six fine-grained urban structure classes + background) segmentation head as our baseline model. The reason is that due to limited computational resources, SegFormer-B0 is the most suitable starting point, and the original head and label set don't match our remapped classes, and a fine-tuned baseline gives a strong, task-aligned reference so any gains can be attributed to our methods rather than simply training the model on the target data. The training setup is shown in the code below:
```
model_base = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    num_labels=7,
    ignore_mismatched_sizes=True
)

model_base.cuda()
print(model_base.device)

training_args = TrainingArguments(
    output_dir="./segformer-thin-structures-v2",
    learning_rate=6e-5,
    num_train_epochs=15,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_total_limit=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    remove_unused_columns=False,
    dataloader_num_workers=8,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=1,
    report_to="none",
)

trainer = Trainer(
    model=model_base,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
print(training_args.device)

trainer.train()
trainer.save_model("./segformer-thin-structures-v2-final")
print("Done!")
```
All methods we explored in the project will use same set of training_args as shown above for better comparison.

The performance of the baseline is shown in the table below:

| class                | Fine-tuned    |
| :---                 |     ---: |
| fence                | 0.3438            |
| car                  | 0.8964        |
| vegetation           | 0.8968        |
| pole                 | 0.3119        |
| traffic sign         | 0.5653        |
| traffic light        | 0.4583        |
| mIoU                 | 0.5788        |

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
