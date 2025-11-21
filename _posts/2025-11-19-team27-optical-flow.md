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

-  Motivation & Improvements
-  Architecture/Methodology
-  Strengths & Limitations

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
