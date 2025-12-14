---
layout: post
comments: true
title: Project Track Project 44 - Streetview Semantics 
author: Benjamin Man, Nathan Leobandung, Jason Jiang, Steven Pan
date: 2025-12-13
---


> Street-level semantic segmentation is a core capability for autonomous driving systems, yet performance is often dominated by severe class imbalance where large categories such as roads and skies overwhelm safety-critical but rare classes like bikes, motorcycles, and poles. Using the BDD100K dataset, this study systematically examines how architectural choices, loss design, and training strategies affect segmentation quality beyond misleading pixel-level accuracy. Starting from a DeepLabV3-ResNet50 baseline, we demonstrate that high pixel accuracy (~ 94%) can coincide with extremely poor mIoU (~ 4%) under imbalance. We then introduce class-weighted and combined Dice-Cross-Entropy/Focal losses, auxiliary supervision, differential learning rates, and gradient clipping, achieving a 10x improvement in mIoU. Finally, we propose a targeted optimization strategy that remaps the task to six safety-critical small classes and leverages higher resolution, aggressive augmentation, and boosted class weights for thin and small objects. This approach significantly improves IoU for bicycles, motorcycles, and poles, highlighting practical trade-offs between accuracy, resolution, and computational cost. However, such increases in resolution resulted in significant increases to training time per epoch, resulting in less training. Overall, the work provides an empirically grounded blueprint for addressing class imbalance and small-object segmentation in urban scene understanding.


* [Streetview Semantics](#streetview-semantics)
  * [What is streetview semantics?](#what-is-streetview-semantics)
  * [Dataset](#dataset)
  * [Initial baseline implementation](#initial-baseline-implementation)
  * [Improvements atop Baseline Model](#improvements-atop-baseline-model)
  * [Further Model Development](#further-model-development)
  * [Model Development Versions](#model-development-versions)
  * [Future uses](#future-uses)
  * [Conclusion](#conclusion)
  * [References](#references)




# Streetview Semantics


## What is streetview semantics?


Streetview semantics involves assigning meaningful labels such as *road*, *sidewalk*, *building*, *vehicle*, *pedestrian*, or *sky* to pixels or regions in images captured from a street-level perspective.


Semantic segmentation of streetview images can be used to transform raw visual input into a structured representation, where each region corresponds to an object or surface with a specific functional role. This semantic understanding enables higher-level reasoning about the environment, such as identifying drivable areas, detecting obstacles, and interpreting relationships between objects in down stream tasks, like autonomus driving.


## Dataset


The BDD100K dataset [2] includes a diverse collection of streetview images designed for autonomus driving research. The dataset includes images captured by a camera mounted on a vehicle from a variety of locations like New York and San Francisco.


We use 10,000 images and their semantically segmented labeled masks to train and test our model.


![A sample image and mask]({{ '/assets/images/44/bddsample.png' | relative_url }})
*A sample image and its corresponding mask from the dataset.*


| Class           | Pixel Count | Percentage |
|-----------------|-------------|------------|
| Road            | 14,112,633  | 26.43%     |
| Sidewalk        | 1,072,369   | 2.01%      |
| Building        | 8,773,136   | 16.43%     |
| Wall            | 300,296     | 0.56%      |
| Fence           | 520,512     | 0.97%      |
| Pole            | 635,225     | 1.19%      |
| Traffic light   | 81,342      | 0.15%      |
| Traffic sign    | 284,603     | 0.53%      |
| Vegetation      | 8,817,392   | 16.51%     |
| Terrain         | 725,905     | 1.36%      |
| Sky             | 11,134,137  | 20.85%     |
| Person          | 187,644     | 0.35%      |
| Rider           | 16,331      | 0.03%      |
| Car             | 5,684,376   | 10.65%     |
| Truck           | 622,560     | 1.17%      |
| Bus             | 401,880     | 0.75%      |
| Train           | 8,826       | 0.02%      |
| Motorcycle      | 7,140       | 0.01%      |
| Bicycle         | 10,650      | 0.02%      |




**Table 1:** Pixel distribution across semantic classes in the dataset.


The BDD100K dataset exhibits severe class imbalance, with a small number of large classes; most notably road, vegetation, and sky-dominating pixel coverage, while many semantically and operationally important classes occupy only a minimal fraction of the image area. This imbalance induces a common failure mode in semantic segmentation, whereby models achieve high pixel-level accuracy by overpredicting majority classes while largely ignoring rare objects. Such behavior is particularly problematic in autonomous driving contexts, where small and thin objects such as bicycles and poles are both visually challenging and safety-critical. Accordingly, our objective is not to optimize global accuracy, but to improve the reliable detection and segmentation of minority classes relative to a baseline, even when this entails trade-offs on dominant classes.




## Initial baseline implementation


We first started with simply feeding our test set to DeepLabV3 [1] with a ResNet-50 Backbone [5].

We chose this model for its proven performance on urban scene understanding (Cityscapes [6], COCO), its Atrous Spatial Pyramid Pooling (ASPP) module for multi-scale context, pretrained ResNet-50 backbone for strong feature representations, and auxiliary classifier head for improved gradient flow.

Our initial model performed deceptively well (when it was in fact not performant at all) because we used pixel-level accuracy as our metric. When checking model predictions on the test set, we can see the model is not performing well:


![Bad segmentation from the old model]({{ '/assets/images/44/badseg.png' | relative_url }})


**High pixel accuracy**: ~ 93%


**Very poor mIoU**: ~ 4%


This was likely caused by a severe class imbalance in the BDD100K dataset. The model learns to predict only common classes, causing high pixel accuracy but poor mIoU.


## Improvements atop Baseline Model


We implemented a variety of additions to our baseline model in hopes of pushing the model to classify low-density classes more.


### 1. **Class Weighting**


We assign higher weight to rarer classes (less pixels labeled in pictures in the dataset) in our loss function in order to get our model to focus on these fine grained details. This prevents the model from straight out ignoring minority classes and simply labeling majority classes to everything. This works as a sort of increased penalty for cheating.


We assign inverse-frequency weight for class $$ f_c $$ and normalize.


$$ w_c = \frac{\frac{1}{f_cC}}{\sum_{i=1}^C w_i}C $$




### 2. **Combined Loss Function**


We use a combined **Dice loss** and **Cross Entropy loss** to better handle the class imbalance and improve boundary predictions.


Dice loss is defined as:


$$ {L}_{d} =1 - \frac{2\sum_{i=1}^n y_{pred_i}y_{true_i} + \epsilon}{\sum_{i=1}^n y_{pred_i} + \sum_{i=1}^n y_{true_i} + \epsilon} $$


Cross Entropy loss is defined as:


$$ {L}_{CE} = -\sum_{c=1}^N y_clog(p_i) $$


The combined loss is simply


$$ {L} = {L}_{d} + {L}_{CE} $$


This creates a training objective that balances per-pixel classification accuracy and region level overlap, encouraging accurate semantic prediction.


### 3. **Auxiliary Loss Usage**


We make use of the two predictions the network produces: full-resolution prediction and the intermediate `aux_out` prediction from an earlier layer. This helps our deep network's gradients have a shorter path to reach earlier layers.


```python
main_loss = criterion(main_out, masks)
aux_loss = criterion_aux(aux_out, masks)
total_loss = main_loss + 0.4 * aux_loss
```


### 4. **Separate Learning Rates**

| Parameter Group       | Learning Rate     | Weight Decay     |
|-----------------------|-------------------|-------------------|
| Backbone Parameters   | 1e-4              | 1e-4              |
| Classifier Parameters | 1e-3              | 1e-4              |




We reduce the learning rate of the pretrained backbone and increase it for the classifier head to allow for better retainment of high-level vision understanding while better learning our dataset.


### 5. **Gradient Clipping**


We clip gradients to a max norm of 1 to prevent exploding gradients. This allows for more stable training.


$$ g_i \leftarrow g_i \frac{1}{||\textbf{g}||_2} $$


## Further Model Development


The initial results of running our model through 6400 training samples on 9 epochs produced best model at the 9th epoch with validation mIoU (on 1600 test samples) of **41.56%**.


![Training]({{ '/assets/images/44/mv1_training.png' | relative_url }})
*Note that the Epoch axies are wrong here; we trained 4 extra epochs after it was already trained for 5.*


This shows promising improvement over our baseline. Looking at some model sample outputs, we see the model is able to identify large segments but struggles to match a precise boundary.


![Some samples]({{ '/assets/images/44/okseg.png' | relative_url }})


This is reflected in our model's per-class mIoU:


| Class          | IoU    |
|----------------|--------|
| road           | 0.6055 |
| sidewalk       | 0.4564 |
| building       | 0.6682 |
| wall           | 0.2186 |
| fence          | 0.3026 |
| pole           | 0.2435 |
| traffic_light  | 0.3113 |
| traffic_sign   | 0.3164 |
| vegetation     | 0.7488 |
| terrain        | 0.3584 |
| sky            | 0.8786 |
| person         | 0.4281 |
| rider          | 0.2265 |
| car            | 0.7943 |
| truck          | 0.4343 |
| bus            | 0.5171 |
| train          | 0.0010 |
| motorcycle     | 0.2024 |
| bicycle        | 0.1842 |

mIoU: 0.4156


In order to test whether our model can effectively differentiate difficult classes, however, we decided to repeat the same training setup using a modified dataset. In this variant, all classes are assigned a weight of 0 except for *pole, vegetation, person, car, motorcycle,* and *bicycle*. These classes are suspected to be especially difficult to segment due to their low frequency in the dataset and, for some, their thin or fine-grained structure.


The resulting model performed **significantly worse overall**, with particularly poor performance on thin objects such as **poles**, **motorcycles**, and **bicycles** with individual IoUs of less than 0.001 and an overall mIoU of 25%.

This collapse in our first 6-class attempt pushed us to design a third version: moving to 1024 x 512 resolution, stronger Focal Loss [3] focus (y = 3.0), heavier class weights (bike/moto 3x, pole 2x), wider scale augmentation (0.5 - 2.0), tighter crops (0.6 - 1.0) to keep small objects in frame, and light random erasing to cut background reliance. The combined Focal Loss [3] and Dice Loss [4] formulation helped target hard examples and improve boundary predictions for thin objects. The goal was to recover and improve thin-object performance, especially for bicycles and motorcycles, while staying focused on the six safety-critical classes. However, these improvements, especially the increase in resolution did lead to a significant increase in training time (one hour per epoch) as we would need to half the batch size.

After 10 epochs with the improved v3 setup, the 6-class evaluation produced the following class-wise weights and IoU scores:

| Class       | Weight | IoU    |
|-------------|--------|--------|
| pole        | 0.233  | 0.0105 |
| vegetation  | 0.100  | 0.6153 |
| person      | 0.817  | 0.4454 |
| car         | 0.100  | 0.7371 |
| motorcycle  | 10.000 | 0.2537 |
| bicycle     | 7.243  | 0.0934 |

mIoU: 0.3592

Although our overall mIoU was still lower than our model for all classes, it still showed significant improvement as not only were pole and bicycle able to break the 0.01 value of IoU, motorcycle improved drastically , reaching an IoU of 0.25. However we do believe that further training could generate even stronger results.

## Model Development Versions


### Model v1 - Baseline DeepLabV3


**Architecture:** DeepLabV3-ResNet50 pretrained on COCO


**Configuration:**
- Resolution: 512 x 256
- Classes: All 19 BDD100K classes
- Loss: Cross Entropy only
- Batch Size: 8


**Performance:**
- Pixel Accuracy: ~ 94%
- mIoU: ~ 4%


**Key Problem:** Severe class imbalance - road and sky dominate


---


### Model v2 - Class Balancing & Loss Improvements


**Improvements over v1:**
1. Class weighting
2. Combined Focal Loss
3. Auxiliary loss branch
4. Separate learning rates
5. Gradient clipping


**Configuration:**
- Resolution: 512 x 256
- Classes: All 19 BDD100K classes
- Loss: Focal (y = 2.0) + Dice (weight = 0.3)
- Batch Size: 8
- Epochs: 10


**Performance on all 19 classes:**
- Pixel Accuracy: ~ 88 - 92%
- mIoU: ~ 40% 
- Common:  ~ 80%
- Medium:  ~ 50%
- Rare:    ~ 20%

**Performance on just 6 classes:**
- mIoU: ~ 20%
- Bicycles, Motorcycles, Poles: ~1%



### Model v3 - Bicycle/Motorcycle Targeted Optimization


**Approach:** Focused optimization on 6 safety-critical urban classes with emphasis on thin/small objects



**Key Improvements over v2:**


1. **Doubled Resolution**
  - **1024 x 512** (4x pixels)
  - Critical for thin objects (bicycles, motorcycles, poles)
  - Improved small object segmentation


2. **Enhanced Focal Loss**
  - Gamma: **y = 3.0**
  - Stronger focus on hard negatives
  - Better handles misclassified edge pixels


3. **Class Weight Boosting**
  - Bicycle: **3.0x** base weight
  - Motorcycle: **3.0x** base weight
  - Pole: **2.0x** base weight
  - Prevents model from ignoring safety-critical minority classes


4. **Wider Scale + Object-Centric Crops**
  - Scale range: **0.5 - 2.0x**
  - Random crop **0.6 - 1.0x** to keep small/thin objects in view
  - Gaussian blur, color jitter for robustness


5. **Combined Loss Refinement**
  - Focal Loss (y = 3.0): Hard example mining
  - Dice Loss (weight = 0.3): Boundary prediction
  - Complementary benefits for small objects


7. **Light Random Erasing (train only)**
  - p = 0.3, small holes to reduce background reliance
  - Encourages robustness to occlusion and clutter


**Configuration:**
- Resolution: **1024 x 512**
- Classes: **6-class subset** (bicycle, motorcycle, pole, car, person, vegetation)
- Loss: Focal (y = 3.0) + Dice (weight = 0.3)
- Batch Size: **4**
- Epochs: 10

**Performance:**
- mIoU: ~ 36%
- pole: ~ 1%
- vegetation: ~ 62%
- person: ~ 45%
- car: ~ 74%
- motorcycle: ~ 25%
- bicycle: ~ 9%


## Future uses

Beyond standard IoU metrics, we implemented a `RoadSafetyAnalyzer` class in our notebook that computes a composite safety score (0 - 100) based on segmentation predictions. This scorer evaluates road visibility, pedestrian and vehicle presence, obstacle detection, and lane clarity to provide a risk-level assessment (LOW, MODERATE, HIGH, CRITICAL) for each scene.

![Best and worst safety scores]({{ '/assets/images/44/safety_comparison.png' | relative_url }})

 Future work could extend this framework to temporal analysis across video sequences, multi-sensor fusion, and adaptive policy generation based on safety scores.

## Conclusion


This work demonstrates that semantic segmentation performance in urban driving scenes cannot be meaningfully assessed, or improved, using pixel-level accuracy alone in the presence of extreme class imbalance. Through a series of controlled model iterations on the BDD100K dataset, we show that loss design, class-aware weighting, and training strategy play a decisive role in recovering minority-class performance. By progressively shifting from a general-purpose baseline to a targeted, high-resolution optimization focused on safety-critical classes, we achieve substantial gains in mIoU and per-class IoU for thin and rare objects such as bicycles, motorcycles, and poles. These results highlight the necessity of task and risk-aware model design in autonomous driving and provide a practical blueprint for addressing imbalance-driven failure modes without reliance on dataset expansion.

## References

[1] Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. "Rethinking atrous convolution for semantic image segmentation." *arXiv preprint arXiv:1706.05587*. 2017.

[2] Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., ... & Darrell, T. "BDD100K: A diverse driving dataset for heterogeneous multitask learning." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2020.

[3] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollar, P. "Focal loss for dense object detection." *Proceedings of the IEEE international conference on computer vision*. 2017.

[4] Milletari, F., Navab, N., & Ahmadi, S. A. "V-net: Fully convolutional neural networks for volumetric medical image segmentation." *2016 fourth international conference on 3D vision (3DV)*. 2016.

[5] He, K., Zhang, X., Ren, S., & Sun, J. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.












