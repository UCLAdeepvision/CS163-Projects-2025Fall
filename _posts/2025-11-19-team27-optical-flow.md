---
layout: post
comments: true
title: Optical Flow
author: Ryan Carney, Phi Nguyen, Nikolas Rodriguez
date: 2025-11-19
---

> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.

<!--more-->

{: class="table-of-content"}

-  TOC
   {:toc}

## 1. Introduction

-  What is Optical Flow?
-  Problem Formalization & Evaluation Metrics
-  The Ground Truth Problem: Why Unsupervised?
-  Report Scope

## 2. Background & Timeline

### Progress and Breakthrough

Prior to the deep learning era, optical flow estimation, which calculates a pixel-wise displacement field between two images , was initially formulated as a continuous optimization problem. The foundational work in this domain was done by Horn and Schunck. Later methods, like DeepFlow (2013), blended classical techniques such as a matching algorithm (that correlates multi-scale patches) with a variational approach to better handle large displacements.

In the era of DL, it showed it could bypass the need to formulate an explicit optimization problem and train a network to directly predict flow. This revolution began with FlowNet (2015), the first CNN model to solve optical flow as a supervised learning task. FlowNet used a generic CNN architecture with an added layer that represented the correlation between feature vectors. Following this, models like FlowNet2 (2017) improved performance by stacking multiple FlowNet architectures but required a large number of parameters (over 160M). This led to a subsequent focus on creating smaller, more efficient models, such as SpyNet (2016), which was 96% smaller than FlowNet , and PWC-Net (2018), which was 18 times smaller than FlowNet2. The current state-of-the-art model is RAFT (Recurrent All-Pairs Field Transforms) (2020), which combines CNN and RNN architectures, updates a single high-resolution flow field iteratively, and ties weights across iterations.

### Datasets

Optical flow models heavily utilize large, synthetically generated datasets. Key benchmarks include: KITTI 2015, which comprises dynamic street scenes captured by autonomous vehicles; MPI Sintel, derived from a 3D animated film, which provides dense ground truth for long sequences with large motions; and the purely synthetic datasets Flying Chairs and Flying Things3D, which trade realism for quantity and arbitrary amounts of samples.

<center>

<img src="{{ site.baseurl }}/assets/images/team27/datasets_sizes.png" />

</center>

### Scarcity of Real-world Data

One of the core limitations in optical flow research is the difficulty of collecting real-world optical flow ground truth data. Due to this, the domain lacks any large, supervised real-world datasets. Optical flow models often have to rely on synthetic data, which are simulated by computers. However, this data is too "clean', in which it doesn't have the natural artifacts that real-world data has such as motion blur, noise, etc. Because of this, models trained on synthetic data alone may not be well-equipped to handle the noisy data in real life. Later on, we will discuss an alternative to supervised learning as a way to combat this data scarcity problem.

## 3. FlowNet: Pioneering Deep Learning for Optical Flow

-  Motivation
-  Architecture Overview
-  FlowNetSimple vs FlowNetCorrelated
-  Correlation Layer Innovation
-  Supervised Training Strategy
-  Strengths & Limitations

## 4. RAFT: Optical Flow Models and Training Techniques in Data-Constrained Environment

### RAFT

RAFT (Recurrent All-Pairs Field Transforms) is a state-of-the-art optical flow model that combines CNN and RNN architectures. It consists of three main components:

1. A feature encoder that generates per-pixel feature vectors for both images, along with a context encoder that processes only the first image ($img_1$).

2. A correlation layer that constructs a 4D correlation volume by taking inner products between all feature pairs, then applies pooling to create multi-scale, lower resolution volumes.

3. A recurrent GRU-based update operator that starts with a zero flow field and iteratively refines it through repeated updates.

<center>

<img src="{{ site.baseurl }}/assets/images/team27/RAFT_arc.png" />

</center>

#### Loss Function

The total loss ($L$) for RAFT-S is defined as the sum of losses on each recurrent block output. This is because the recurrent cell outputs create a sequence of optical flow predictions ($f_1, ..., f_N$).

-  Distance Metric: The loss function for each output is the L1 distance between the ground truth ($gt$) and the upsampled prediction ($f_i$). This is the same distance metric used in the FlowNet loss function.

-  Weighting: The loss for each step in the sequence is weighted by an exponentially decreasing parameter, $\gamma$, where $\gamma=0.8$. This weighting gives more importance to earlier predictions in the sequence.

The total loss is expressed by the following formula:

$$L=\sum_{i=1}^{N}\gamma^{i-N}||gt-f_{i}||_{1} \text{, } \gamma=0.8$$

One major design shift in RAFT is that it keeps a single high-resolution flow field and refines it directly, avoiding the usual coarse-to-fine upsampling strategy seen in earlier models. Its update module is a lightweight recurrent block that reuses the same weights across steps. For efficiency, RAFT-S was used, a smaller variant that reduces parameters by using narrower feature channels, slimmer bottleneck residual units, and a single $3\times3$ convolution GRU cell.

### FlowNet Modification (Deconvolutional Upsampling)

DU was introduced as an alternative to the usual last stages of FlowNet, where the decoder output normally goes through another convolution and then a bilinear upsampling step. Instead, it uses a single transposed convolution to both learn richer features and scale the output, aiming to produce flow predictions that are more detailed and accurate.

### Data Augmentations Used

A set of augmentation methods was applied to the synthetic Flying Chairs dataset to help prevent overfitting, even though the dataset is already quite large. The hyperparameters followed those used for the original RAFT training setup, and the flow labels were adjusted after each transformation.

The methods used included:

-  Cropping all images to the same size then normalize the pixel values.
-  Random horizontal and vertical flips.
-  Random stretching to simulate zooming in or out.
-  Random asymmetric jitter, which alters brightness, contrast, saturation and hue.
-  Random erasing of certain regions in the second image ($img_2$) of each pair.

| Flying Chairs before augmentation                                             | Flying Chairs after augmentation                                                |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| <img src="{{ site.baseurl }}/assets/images/team27/FC_no_aug.png" width="400"> | <img src="{{ site.baseurl }}/assets/images/team27/FC_with_aug.png" width="400"> |

### Evaluation Metrics

The primary metric was End-Point Error (EPE), which measures the Euclidean distance between the predicted flow vector and the ground-truth vector at each pixel:

$$\text{EPE} = \sqrt{(\Delta x_{gt} - \Delta x_{pred})^{2} + (\Delta y_{gt} - \Delta y_{pred})^{2}}$$

Lower EPE values indicate more accurate flow estimates. We also report the percentage of pixels with errors below 1, 3, and 5 pixels (1px, 3px, 5px), which helps measure how well the model handles both small and moderate motions. Finally, we include the F1 outlier rate, defined as the percentage of pixels whose EPE exceeds both 3 pixels and 50% of the magnitude of the ground-truth flow. These metrics together allow us to compare overall accuracy, motion sensitivity, and robustness to outliers before analyzing model performance.

### Results

The experiments compared FlowNet, the modified FlowNet with deconvolutional upsampling (DU), and RAFT-S across different amounts of pre-training data (0, 500, and 2000 Flying Chairs pairs) and with or without data augmentation. All models were fine-tuned on 520 Sintel image pairs and evaluated on both Sintel and KITTI. Overall, the results showed that dataset size and augmentation played a major role in model performance, especially in low-data settings, and that RAFT benefited significantly more from augmentations than FlowNet.

Across nearly all settings, increasing the number of Flying Chairs samples improved performance. On Sintel, FlowNet’s EPE dropped from 10.18 → 6.79 as pre-training rose from 0 to 2000 samples, and RAFT showed an even steeper improvement, especially when paired with augmentation (e.g., EPE 18.37 → 6.51 at 2000 samples). However, results on KITTI revealed that FlowNet did not always benefit from larger synthetic datasets, likely due to the domain gap between Flying Chairs and real scenes in KITTI, suggesting that mismatched pre-training can sometimes hurt generalization.

| Flownet on Sintel                                                                     | Flownet on KITTI                                                                     |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| <img src="{{ site.baseurl }}/assets/images/team27/flownet_on_sintel.png" width="400"> | <img src="{{ site.baseurl }}/assets/images/team27/flownet_on_kitti.png" width="400"> |

Data augmentation consistently helped when the test domain differed from the training domain. For RAFT in particular, augmentation sharply reduced error. For example, on Sintel with 2000 pre-training samples, EPE improved from 38.78 (no aug) to 6.51 (with aug). FlowNet saw smaller gains, and in some Sintel cases the improvement was minimal, showing that FlowNet tends to overfit more regardless of augmentation.

| RAFT on Sintel                                                                     | RAFT on KITTI                                                                     |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| <img src="{{ site.baseurl }}/assets/images/team27/raft_on_sintel.png" width="400"> | <img src="{{ site.baseurl }}/assets/images/team27/raft_on_kitti.png" width="400"> |

The modified FlowNet architecture with deconvolutional upsampling performed almost identically to the original. For instance, at 2000 pre-training samples on Sintel, the new model achieved an EPE of 6.80, essentially the same as the original 6.79. The only noticeable pattern was that DU gave a slight advantage when little or no pre-training data was available, while the original FlowNet recovered a small edge with larger datasets. Overall, the architecture change did not introduce meaningful improvements.

<center>

<img src="{{ site.baseurl }}/assets/images/team27/modded_flownet.png" width="700">

</center>

Qualitatively, RAFT produced cleaner and more consistent flow fields than FlowNet, especially when pre-training and augmentation were combined. In scenes from the Driving and FlyingThings, RAFT captured fine-grained motion and global consistency better, while FlowNet focused more on object boundaries and sometimes missed motion details. In the low-data setting with no pre-training, RAFT still produced more coherent motion estimates, whereas FlowNet tended to rely heavily on edges rather than true frame-to-frame displacement.

| RAFT on Sintel (no pre-training)                                                      | Flownet on Sintel (no pre-training)                                                      |
| ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| <img src="{{ site.baseurl }}/assets/images/team27/raft_preds_sintel.png" width="400"> | <img src="{{ site.baseurl }}/assets/images/team27/flownet_preds_sintel.png" width="400"> |

## 5. UFlow: What Matters in Unsupervised Optical Flow

-  Motivation: The Supervised Learning Bottleneck
-  Key Research Questions
-  Architecture & Training Components
-  Ablation Studies: What Actually Matters?
-  Strengths & Limitations

## 6. Comparative Analysis

-  Supervised vs Unsupervised: Trade-offs
-  Performance Comparison Across Datasets
-  Training Data Requirements
-  Generalization Capabilities
-  Computational Costs
-  Practical Considerations

## 7. Discussion

-  Evolution of the Field
-  When to Use Each Approach
-  Open Challenges
-  Future Directions

## 8. Conclusion

## Reference

Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2016.

---
