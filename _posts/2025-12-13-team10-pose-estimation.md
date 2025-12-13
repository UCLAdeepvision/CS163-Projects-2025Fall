---
layout: post
comments: true
title: Introduction to Camera Pose Estimation
author: Team 10
date: 2025-12-13
---

> Camera pose estimation is one important component in computer vision used for robotics, AR/VR, 3D reconstruction, and more. It involves determining the camera’s 3D position and orientation, also known as the “pose” in various environments and scenes. PoseNet, MeNet, and JOG3R are all various deep learning techniques used to accomplish camera pose estimation. There are also sensor-based tracking like LEDs and particle filters. We focus on geometric methods, specifically Structure-from-Motion (SfM).

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}


## Introduction to Camera Geometry Basics

Camera geometry is necessary to map out the relationship between 3D points in the real world and the corresponding 2D projection which are pixel coordinates. There are a few technical terms we will use. First, the process of projecting 3D coordinates to 2D is typically modeled by the Pinhole Camera Model as shown in the figure below. This model describes the mathematical relationship of the projection where the pinhole is the center of projection and the origin of a Euclidean coordinate system and the Z axis is the image plane or focal plane. [number]

Figure 1. Pinhole Camera Model. [number].

Second, we define world coordinates as representations of a location in the World Cartesian coordinate system. The 3D position of a point P in this world is set to P = (X, Y, Z). Second, camera coordinates represent the coordinate system in the virtual camera plane. The planes x and y are parallel to the world coordinates axes X and Y but run in opposite directions. The signs of the coordinates will stay the same meaning both images in the world and camera coordinates are identical except for a flip. Assuming the distance from the pinhole to the sensing plane is represented by f also known as the focal length, then we can define the following equations:

Figure 2. Perspective projection equations. [number]

Based on the perspective projection equations which are from the notion of similar triangles, we can map the 3D point [X, Y, Z]T to the coordinates [x, y]T on the image plane. We also add a fourth coordinate to get P = (X, Y, Z, 1)T to get homogeneous coordinates. We can confirm that we get the perspective projection if we multiply P with the matrix K as shown below. [number]

Figure 3. Perspective projection of 3D point p. [number]

Then we can transform the homogeneous coordinates scaled by the camera’s focal length f to convert it to heterogeneous coordinates and divide by Z into 2D coordinates. As a result, the 2D projected coordinates can be labeled as p.

## Structure-from-Motion Overview

SfM is the geometric method of recovering the 3D structure and camera motion through a collection of images which addresses a fundamental problem in computer vision. SfMs are applied to novel view synthesis which generates images of a specific scene or subject from a certain point of view given that the only available information of the specific scene or subject is images taken from different points of view than the one we are trying to synthesize. SfMs are also used for cloud-based mapping and localization which involves using powerful cloud servers for generating dynamic mappings of surrounding environments often used for autonomous vehicles and robots.

Figure 4. Structure from Motion point cloud. [number]

Sfm works by analyzing how various points in the images shift and change relative to one another as the camera angle and position changes. This is typically done through motion parallax which reconstructs the scene or subject’s 3D structure and camera orientation and position for each image taken at different point of views. Going more in detail, the first step involves taking multiple overlapping photos to capture either a scene or subject. Photos must be taken in well-lit lighting conditions with all features visible in multiple photos and no reflective surfaces. Then, software processes the photos to distinguish individual features such as textures and corners. Then, the software analyzes the 2D movement of all these features across all the images with the algorithm inferring the camera’s movement and the 3D position of the features. This perception is similar to how humans perceive depth such as how objects closer to the observer moves faster than objects further away move. Finally, the 3D points are reconstructed to build a 3D model that is typically in the form of point clouds. [number]

## Methods

There are two methods of SfM: incremental and global. Incremental SfM starts with a pair of images and creates a 3D model by adding cameras and points step by step to estimate a similar pose of the existing construction. Then, it seeds the pair selection to form an initial reconstruction also known as triangulation. This creates a robust and more accurate model. On the other hand, global SfM considers all images and their relationship with each other to be a single optimization problem and attempts to reconstruct the camera poses and points simultaneously. First there is a feature mapping process for all the images that then builds a view graph. Finally, through various rotation and translation averaging, we get a global optimization and a 3D model. Global SfM is more scalable and efficient for large scale mapping but is not as accurate than incremental nor is it as robust if there are outliers or missing data such as if the graph is not fully connected.

Figure 5. Incremental SfM vs. Global SfM. [number]

## GLOMAP

### GLOMAP Overview

GLOMAP is a recent global SfM system by Pan et al. that addresses challenges encountered in global SfM. At a high level, GLOMAP is a general-purpose pipeline designed to improve the accuracy and robustness of global SfMs while retaining its scalable and efficient characteristics of global SfM. In fact, Pan et. al report that GLOMAP produces on par with or even superior to COLMAP (the incremental gold standard) in accuracy, yet is orders of magnitude faster. For example, on a large-scale dataset LIN from LaMAR of 36,000 images, GLOMAP obtained ~90% correct camera pose recall at 1m error and reconstructed the scene in approximately 5.5 hours, while COLMAP had only ~50% recall and took more than 7 days to process the same data. These results highlight a significant step forward for global SfM methods.

### Challenges of Global SfM

The key difficulty in global SfM lies in how it handles noise and outliers when combining all pairwise relations simultaneously. In particular, the global translation averaging step is the main cause of global SfM’s shortfall in robustness. After estimating relative motions between image pairs (the view graph), global SfM does rotational averaging, which estimates all camera orientations from pairwise relative rotation. To obtain consistent global camera positions from noisy pairwise translations, however, is the more challenging problem.

When estimating motion between two images, only the direction of the translation can be recovered and not the distance. This means that every pair of cameras have an unknown scaling factor. This introduces scale ambiguity that requires multiple loops or triplets of views to resolve. Furthermore, if those triplets form “skewed” geometric configurations, the scale estimates become very noise-sensitive. [2]

Another challenge is the global SfM’s heavy dependence on camera intrinsics. This information is needed so that global SfM can decompose two-view geometry into a direction and scale, and if there is any calibration error, the translation direction becomes highly inaccurate. [1]

In scenarios with nearly collinear motion (e.g. walking forward with a front-facing phone, driving down a street, etc.) the geometry does not provide enough information for translation averaging, and this can cause reconstructions to become unreliable. [2]

### GLOMAP Contributions

The key innovation introduced by GLOMAP is jointly estimating all camera positions and 3D points, which replaces the translation averaging step. In other words, instead of first computing camera translations in isolation and then triangulating points, GLOMAP merges these into one optimization problem in a step the authors call global positioning. This approach is fundamentally different from prior global SfM systems, which typically did a rotation averaging step, followed by translation averaging, and then triangulation. GLOMAP’s joint global position step replaces the fragile process with one unified optimization that solves for camera positions and 3D points at the same time.

![Figure 6]({{ '/assets/images/team10/glomap_table.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
Figure 6. Performance of GLOMAP on ETH3D MVS Benchmark [1]

GLOMAP represents a step forward in balancing accuracy, speed, and robustness in Structure-from-Motion. While many traditional systems excel in one or two of these areas, GLOMAP achieves strong performance across all three. It avoids reliance on carefully selected initialization pairs, handles large-scale reconstructions efficiently, and remains robust under noisy conditions.

## Visual Geometry Grounded Deep SfM

### VGGSfM Overview

VGGSfM is a deep learning-based SfM pipeline designed to be fully differentiable end-to-end. It is conceptually closer to a global SfM approach since it estimates all camera poses jointly, but it avoids the typical weaknesses of classical global SfM such as noise-sensitive averaging of rotations and translations.

### Challenges with Current Incremental SfM Systems

Traditional incremental SfM systems, like COLMAP, build up 3D reconstructions one step at a time starting with a good image pair and then gradually adding new views. While this works well in many cases, it also introduces a number of weaknesses.

### VGGSfM Contributions

A major innovation in VGGSfM is its approach to correspondence tracking. Traditional SfM pipelines typically rely on chaining two-view matches across image pairs to form multi-view feature tracks.

Figure 7. Performance of GLOMAP on ETH3D MVS Benchmark [3]

## InstantSfM

### InstantSfM Overview

With different SfMs comes with different CPU computational costs that InstantSfM aims to decrease.

### InstantSfM Contributions

Figure 8. The Levenberg-Marquardt Algorithm

## Conclusion

Camera pose estimation plays an important role in creating 3D reconstructions from video and image data.

## Sources

[1] Pan, Linfei, et al. “Global Structure-from-Motion Revisited.” ECCV, 2024.  
[2] Tao, Jianhao, et al. “Revisiting Global Translation Estimation with Feature Tracks.” CVPR, 2024.  
[3] Qian, Yiming, et al. “Visual Geometry Grounded Structure from Motion.” arXiv:2312.04563, 2023.
