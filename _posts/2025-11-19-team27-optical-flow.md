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

The problem of optical flow in computer vision is simply the estimation motion in images, represented as a flow field at the pixel level. Being able to effectively solve this problem is crucial to many real-world applications, such as autonomous driving, object tracking, and general robotics. While the problem statement may be simple, there are various factors which make this a challenging task in practice. As such, the problem of optical flow is a fundamental one in the field of computer vision.

Given how complex the issue of optical flow is, there have been many different attempts, spanning multiple machine learning paradigms, in an attempt to solve it. Traditionally, researchers in this field have taken a supervised learning approach, commonly with convolutional neural networks, to solve this problem. Most notably, the FlowNet model, along with the RAFT architecture, are among the state of the art in optical flow supervised learning. However, the unsupervised learning approach has recently gained traction in this field, largely driven by a global issue in optical flow problem, which is a lack of labeled training data. In fact, the magnitude of this issue is so great that supervised approaches often rely on synthetic data to train the models. Among unsupervised learning models, UFlow is the most notable.

The scope of this report is to take a deep dive into each of these approaches. This is accomplished by first exploring each of these models individually, and then completing an analysis of how they compare. The individual descriptions include the relevant architectures, training methodologies, and other uniqueness factors. The final analysis and discussion will not only display the strengths and weaknesses of each approach, but also provide a timeline for how this field has progressed over time. The benchmarks and datasets used to compare these approaches are the standards in the optical flow field, including the endpoint error metric and the Sintel and KITTI datasets. Further, the applicable use cases for each of these approaches will also be discussed.

## 2. Background & Timeline

### Progress and Breakthrough

Prior to the deep learning era, optical flow estimation, which calculates a pixel-wise displacement field between two images , was initially formulated as a continuous optimization problem. The foundational work in this domain was done by Horn and Schunck. Later methods, like DeepFlow (2013), blended classical techniques such as a matching algorithm (that correlates multi-scale patches) with a variational approach to better handle large displacements.

In the era of DL, it showed it could bypass the need to formulate an explicit optimization problem and train a network to directly predict flow. This revolution began with FlowNet (2015), the first CNN model to solve optical flow as a supervised learning task. FlowNet used a generic CNN architecture with an added layer that represented the correlation between feature vectors. Following this, models like FlowNet2 (2017) improved performance by stacking multiple FlowNet architectures but required a large number of parameters (over 160M). This led to a subsequent focus on creating smaller, more efficient models, such as SpyNet (2016), which was 96% smaller than FlowNet , and PWC-Net (2018), which was 18 times smaller than FlowNet2. The current state-of-the-art model is RAFT (Recurrent All-Pairs Field Transforms) (2020), which combines CNN and RNN architectures, updates a single high-resolution flow field iteratively, and ties weights across iterations.

### Datasets

Optical flow models heavily utilize large, synthetically generated datasets. Key benchmarks include: KITTI 2015, which comprises dynamic street scenes captured by autonomous vehicles; MPI Sintel, derived from a 3D animated film, which provides dense ground truth for long sequences with large motions; and the purely synthetic datasets Flying Chairs and Flying Things3D, which trade realism for quantity and arbitrary amounts of samples.

<center>

<img src="{{ site.baseurl }}/assets/images/team27/datasets_sizes.png" />

</center>

_Figure 1. KITTI is the only non-synthetic dataset. Its size compared to other synthetic datasets highlights the scarcity of real-world data [2]._

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
*Figure 1. RAFT model architecture illustrating the feature encoder, correlation volume construction, and recurrent update operator [2].*

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

_Figure 2. Visualization of Flying Chairs samples before and after data augmentation, showing the effects of geometric and photometric transformations used during training [2]._

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

_Figure 3. FlowNet performance on Sintel and KITTI, displaying metrics after fine-tuning under different data conditions [2]._

Data augmentation consistently helped when the test domain differed from the training domain. For RAFT in particular, augmentation sharply reduced error. For example, on Sintel with 2000 pre-training samples, EPE improved from 38.78 (no aug) to 6.51 (with aug). FlowNet saw smaller gains, and in some Sintel cases the improvement was minimal, showing that FlowNet tends to overfit more regardless of augmentation.

| RAFT on Sintel                                                                     | RAFT on KITTI                                                                     |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| <img src="{{ site.baseurl }}/assets/images/team27/raft_on_sintel.png" width="400"> | <img src="{{ site.baseurl }}/assets/images/team27/raft_on_kitti.png" width="400"> |

_Figure 4. RAFT performance on Sintel and KITTI, displaying metrics after fine-tuning under different data conditions [2]._

The modified FlowNet architecture with deconvolutional upsampling performed almost identically to the original. For instance, at 2000 pre-training samples on Sintel, the new model achieved an EPE of 6.80, essentially the same as the original 6.79. The only noticeable pattern was that DU gave a slight advantage when little or no pre-training data was available, while the original FlowNet recovered a small edge with larger datasets. Overall, the architecture change did not introduce meaningful improvements.

<center>

<img src="{{ site.baseurl }}/assets/images/team27/modded_flownet.png" width="700">

</center>

_Figure 5. Predictions from the modified FlowNet model using deconvolutional upsampling, compared against the original architecture on Sintel [2]._

Qualitatively, RAFT produced cleaner and more consistent flow fields than FlowNet, especially when pre-training and augmentation were combined. In scenes from the Driving and FlyingThings, RAFT captured fine-grained motion and global consistency better, while FlowNet focused more on object boundaries and sometimes missed motion details. In the low-data setting with no pre-training, RAFT still produced more coherent motion estimates, whereas FlowNet tended to rely heavily on edges rather than true frame-to-frame displacement.

| RAFT on Sintel (no pre-training)                                                      | Flownet on Sintel (no pre-training)                                                      |
| ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| <img src="{{ site.baseurl }}/assets/images/team27/raft_preds_sintel.png" width="400"> | <img src="{{ site.baseurl }}/assets/images/team27/flownet_preds_sintel.png" width="400"> |

_Figure 6. Comparison of RAFT and FlowNet without pre-training, illustrating RAFT’s stronger inductive bias for motion estimation in extremely low data conditions [2]._

## 5. UFlow: What Matters in Unsupervised Optical Flow

### Motivation: The Supervised Learning Bottleneck

One of the major limitations of supervised models attempting to solve the optical flow problem is the lack of available labeled data. As previously discussed, one approach to mitigating this issue is to generate synthetic data and use it to train the supervised models. While this can be a viable option, it is not without trade offs. Namely, this approach is less effective for objects with less rigid geometry, leading to manual intervention being needed [3].

Thus, an alternative solution to the optical flow problem is to abandon the supervised approach in favor of unsupervised models. This paradigm is particularly enticing because of how plentiful training data becomes, as any unlabeled video could be used. Additionally, unsupervised models typically showed advantages in inference speed and generalization as a result of being trained directly on real world (opposed to synthetic) data [3].

These reasons were the driving force behind the creation of the UFlow model, which was created as a means to determine the most important factors in training an unsupervised model. Among the aspects that were investigated were photometric loss, occlusion handling, and smoothness regularization.

### Architecture & Training Components

The underlying architecture that UFlow is based on is the PWC-Net, which is then improved upon to determine the most important components. Because the pyramid, warping, and cost (PWC) approach was initially intended for supervised learning, adjustments had to be made during the creation of the UFlow model. For example, the highest pyramid level was removed to decrease the model size and dropout was incorporated for additional regularization.

Despite these changes, UFlow at its core uses a self-supervised training loop across five pyramid levels to ultimately predict the flow field from two images, as shown in the figure below. Firstly, a shared CNN is used to extract the features from the two images. These features are then fed into the pyramid, where each level has a loop involving warping, cost volume computation, and flow estimation. The warping layer in each block uses the current flow estimate to align the second image's features with the first, creating the photo-consistency signal, which is needed to define photometric loss, the unsupervised training objective. The architecture behind the cost volume computation is a correlation layer, while the flow estimation at each block is done with a CNN.

<img src="{{ site.baseurl }}/assets/images/team27/wcf-loop.png" />

As previously mentioned, the unsupervised training objective is the photometric loss. While the team behind UFlow experimented with multiple different losses, the main one used was the generalized Charbonnier loss function, given by

$$L_C = \frac{1}{n} \sum \left((I^{(1)} - w(I^{(2)}))^2 + \epsilon^2\right)^\alpha$$

where $$\epsilon = 0.001$$ and $$\alpha = 0.5$$.

### Ablation Studies: What Actually Matters?

With the base model established, the team behind UFlow aimed to determine exactly which components mattered the most in an unsupervised model for optical flow. The major components that were found to have the biggest effects following analysis were occlusion handling, data augmentation, and smoothness regularization.

#### Occlusion Handling

As previously mentioned, the UFlow model takes two images as input, and feeds them into a shared CNN for feature extraction. The issue of occlusions arises when specific features or regions of one image are not present in the partner image. If this is not accounted for and handled correctly, any photometric loss calculations would include garbage values. As such, all occlusions, along with any other pixels deemed invalid, need to somehow be detected and gracefully handled. The approach taken by UFlow is to use a “range map”, which stores information regarding the number of pixels in the partner image that are mapped to each pixel in a target image. This range map is then used to determine which pixels are not mapped to at all in the target image, and those pixels are deemed occlusions. Against the KITTI dataset specifically, the approach taken by UFlow is a forward-backward consistency check, which marks pixels as occlusions whenever the flow and back-projected flow disagree by more than some margin [3]. Thus, these pixels are “masked”, where this mask can be calculated by [THIS EQUATION]. In this latter approach, the team found that stopping the gradient at the occlusion mask improved performance.

#### Data Augmentation Strategies

The next component that was analyzed within UFlow was data augmentation. Some of the data augmentation techniques used in training UFlow included color channel swapping, hue randomization, image flipping, and image cropping. However, the most emphasized augmentation technique during this process was continual self-supervision with image resizing. This specific method helped stabilize the training by starting with downsampled, lower-resolution images and then progressively increasing the input resolution over the training run, hence the resizing. The key result found by the team with respect to this factor was that both color and geometric augmentations were key for unsupervised models achieving good generalization.

#### Smoothness Regularization

In addition to data augmentation, one other component of UFlow that was analyzed was smoothness regularization. This aims to ensure that changes in flow estimations are not abrupt, which can happen for inputs with large homogenous regions. Specifically, the performance of both first-order and second-order smoothness regularization were investigated by the team. The difference between these two is simply which derivative of the estimated flow is being considered, and both can be represented by the equation below. The key finding related to this component was that it is significantly more beneficial to apply this smoothness regularization at the lower resolution of the flow itself opposed to the higher original input image resolution. In regards to the optimal regularization order, there was no answer as to which was definitely better, as the optimal order was different for the different datasets.

$$L_{\text{smooth}}^{(k)} = \frac{1}{n} \sum \exp\left(-\frac{\lambda}{3} \sum_{c} \left|\frac{\partial I_c^{(1,\ell)}}{\partial x}\right|\right) \left|\frac{\partial^k V^{(1,\ell)}}{\partial x^k}\right| + \exp\left(-\frac{\lambda}{3} \sum_{c} \left|\frac{\partial I_c^{(1,\ell)}}{\partial y}\right|\right) \left|\frac{\partial^k V^{(1,\ell)}}{\partial y^k}\right|$$

where $$k$$ denotes the order of smoothness (first-order or second-order), $$c$$ indexes color channels, and $$\lambda$$ controls the edge-aware weighting.

All together, a comparison of the estimated flow without each of these three components can be seen in the figure below.

<img src="{{ site.baseurl }}/assets/images/team27/ablations-comparisons.png" />

### Results And Performance

In order to objectively determine which components of unsupervised models matter the most, the team had to quantitatively measure UFlow’s performance on multiple different datasets. The metrics used to benchmark the model were endpoint error and error rate, which are standard for the KITTI dataset and therefore the optical flow problem as a whole. In addition to KITTI, the Sintel dataset was also used to evaluate UFlow.

In terms of performance, the key takeaways were that UFlow set the new gold standard for unsupervised models, by performing at a similar level to the supervised FlowNet2, despite being completely trained without ground-truth labels. Of course, this was achieved through the careful evaluation, optimization, and combination of the various different components of an unsupervised model, and not a single new “breakthrough” technique.

### Strengths & Limitations

Like any other engineering feat, the success of the UFlow model came with tradeoffs. One major strength of UFlow, and unsupervised learning as a whole in the optical flow problem, is the abundance of training data available for use. The ability to take any color video and train on it makes the unsupervised learning paradigm very appealing in the context of optical flow, especially considering the current lack of labeled data. Further, by training on real-world data opposed to almost exclusively synthetic data, UFlow has shown to achieve superior generalization on real-world inputs. Lastly, in the case of the UFlow model itself, there is clearly a systematic way of determining the importance of each of the major components, setting the foundation for future unsupervised learning models to tackle the optical flow problem.

However, UFlow did show that it has some limitations as well. Firstly, in the case of absolute performance, UFlow was still outperformed by the best fine-tuned supervised models (despite performing on par with popular supervised models like FlowNet2). Further, while it was previously discussed that each of the mentioned components can be tuned for optimal results, this tuning can be expensive in practice. This is because, in the process of training UFlow, the researching team had to train a new model for each ablation study. Lastly, in terms of hardware and physical requirements, the PWC-net used by UFlow can be memory intensive. As such, training an unsupervised model like UFlow typically requires more available memory.

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

We discussed the evolution of the approaches to the optical flow problem over time, and analyzed the tradeoffs associated with the different approaches. We started by analyzing FlowNet, which acted as a pioneer for deep learning in the field of optical flow. The key point here was that FlowNet successfully used Convolutional Neural Networks to address the optical flow problem. Next was RAFT, which maintained the supervised approach, but used a feature encoder, correlation layer, and a recurrent GRU-based update operator as the key components in the solution. Lastly, we analyzed UFlow, which differed greatly from the other two models as UFlow was unsupervised. Comparing these three approaches did not reveal a definitive best model, but instead highlighted a different theme, which was that the optimization of the internal components of these models was more important than the exact training paradigm. This was also a key finding during the ablation studies from the team behind UFlow.

One theme that was apparent and common throughout the different approaches was that a lack of training data was a big motivator behind the methods used. As such, much of the realized success and innovations can be attributed to this one single constraint. For example, the RAFT model highlighted the importance of the data augmentation strategies used, a result achieved directly from the initial lack of sufficient labeled data. In the more extreme case, UFlow completely disregarded the need for labeled data and paired an unsupervised approach with research into which components were the most important. Once again, this innovation was driven from what was initially a major constraint. Clearly, this idea is extremely relevant in the optical flow field, but is also very generalizable to other fields of computer science, which may be currently limited by some constraint.

Clearly, the problem of optical flow is an important one, as there are many optical flow applications such as autonomous driving and robotics. As these fields grow, so will the need to find even better solutions to optical flow. The highlighted approaches show that there are multiple paths forward in this field, and future models will have the benefit of using the key findings from FlowNet, RAFT, and UFlow as the foundation.

## Reference

Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2016.

[2] Isik, Senem, Shawn Zixuan Kang, and Zander Lack. “Optical Flow Models and Training Techniques in Data-Constrained Environment.” CS231N: Convolutional Neural Networks for Visual Recognition, Stanford University. 2022.

[3] Jonschkowski, Rico, et al. "What Matters in Unsupervised Optical Flow." _European Conference on Computer Vision (ECCV)_. 2020.

---
