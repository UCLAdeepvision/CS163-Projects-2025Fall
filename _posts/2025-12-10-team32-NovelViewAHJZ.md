---
layout: post
comments: true
title: "Project Track: Exploring Modern Novel View Generation Methods"
author: Team 32
date: 2025-12-10
---

> <!-- PASTE: Motivation paragraph here as the abstract/intro -->

<!--more-->

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Motivation](#motivation)
- [Current Novel View Generation Methodologies](#current-novel-view-generation-methodologies)
   * [3D Gaussian Splatting](#3d-gaussian-splatting)
      + [Overview](#overview)
      + [Method](#method)
      + [Architecture](#architecture)
      + [Results](#results)
      + [Discussion](#discussion)
      + [Running the Codebase](#running-the-codebase)
         - [Exploring Activation Functions](#exploring-activation-functions)
         - [Clamping](#clamping)
   * [Large View Synthesis Model (LVSM)](#large-view-synthesis-model-lvsm)
      + [Overview](#overview-1)
      + [Method](#method-1)
      + [Architecture](#architecture-1)
      + [Results](#results-1)
      + [Discussion](#discussion-1)
      + [Running the Codebase](#running-the-codebase-1)
- [Conclusion](#conclusion)
- [Reference](#reference)

<!-- TOC end -->

<!--
IMAGE SYNTAX REFERENCE:
![Alt Text]({{ '/assets/images/team32/your_image.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig X. Caption for image* [citation].

TABLE SYNTAX REFERENCE:
|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |

FORMULA SYNTAX REFERENCE:
Block formula: $$ y = mx + b $$
Inline formula: $$y = mx + b$$
-->

## Motivation

Historically, novel view synthesis (NVS) has relied on volumetric radiance field approaches, such as NeRF. While effective, these methods are often computationally expensive to train and prohibitively slow to render for real-time applications. To address these limitations, researchers have developed new architectures that reduce computational costs while maintaining or exceeding visual fidelity. This report examines two distinct solutions to this challenge: 3D Gaussian Splatting (3DGS) and the Large View Synthesis Model (LVSM).

## Current Novel View Generation Methodologies

### 3D Gaussian Splatting

> **Note:** This section provides an overview of 3D Gaussian Splatting based on the original paper by Kerbl et al. [1]. All figures and methodology descriptions are derived from their work.

#### Overview

Following Radiance Field-based methods such as NeRF, researchers identified shortcomings in costly training and subpar quality. Before this paper, unbounded and complete scenes with high-resolution rendering (1080p) were not achieved by any existing methods. This lack of speed and quality made real-time, high-quality rendering seem far-fetched. 3D Gaussian Splatting introduced three key elements that greatly improved visual quality and maintained competitive training times while enabling real-time novel-view synthesis.

1. Represent the scene with 3D Gaussians, which avoids unnecessary computation in empty space in volumetric radiance fields
2. Perform interleaved optimization and density control to achieve an accurate representation of the scene
3. Develop a fast visibility-aware rendering algorithm that accelerates training and allows real-time rendering

#### Method

<!-- PASTE: 3DGS Method content (starts with "Given input images with known camera intrinsics...") -->

Given input images with known camera intrinsics and extrinsics obtained via Structure-from-Motion (SfM), 3D Gaussian Splatting constructs an explicit radiance field representation using a set of anisotropic 3D Gaussian primitives. The goal of this method is to synthesize a novel target image corresponding with a target camera pose.

The Gaussians are defined by a full 3D covariance matrix $$\Sigma$$ in 3D space, centered at a mean point $$\mu$$. The following equation defines how strongly a Gaussian contributes at a given point $$x$$, based on the pixel's distance from the mean.

![G(x) Equation]({{ '/assets/images/32/3dgs_Gx.png' | relative_url }})
{: style="width: 300px; max-width: 100%;"}

Through the blending process, the Gaussian is multiplied by $$\alpha$$. In order to project the 3D Gaussian to a 2D image for rendering, a 2D covariance matrix $$\Sigma'$$ must be calculated to define the ellipse shape of the Gaussian splat in image space. $$J$$ represents the Jacobian of the affine approximation of the projective transformation, while $$W$$ applies the world-to-camera transform to the covariance, effectively handling the rotation of the camera extrinsics for the novel view.

![Sigma Prime Equation]({{ '/assets/images/32/3dgs_jacobian.png' | relative_url }})
{: style="width: 300px; max-width: 100%;"}

Optimizing the covariance matrix $$\Sigma$$ to obtain 3D Gaussians that represent the radiance field is the right idea. However, covariance matrices must be symmetric and positive semi-definite to have meaning in physical space, and gradient updates could easily lead to the matrix becoming invalid. Instead, by defining the matrix using a scaling matrix $$S$$ and rotation matrix $$R$$, the corresponding $$\Sigma$$ can be found. These parameters can be trained in such a way that preserves the positive semi-definite, symmetric nature of $$\Sigma$$.

![Sigma Decomposition]({{ '/assets/images/32/3dgs_sigma_parameterized.png' | relative_url }})
{: style="width: 300px; max-width: 100%;"}

By storing both factors separately, a 3D vector $$\mathbf{s}$$ for scaling and a quaternion $$\mathbf{q}$$ for rotation, they can be trivially converted to their respective matrices by normalizing $$\mathbf{q}$$ to obtain a valid unit quaternion. Stochastic Gradient Descent is used for optimization, with a loss function combining $$L_1$$ and a D-SSIM term for photometric loss.

![Loss Function]({{ '/assets/images/32/3dgs_loss_function.png' | relative_url }})
{: style="width: 300px; max-width: 100%;"}

A key aspect of capturing scene complexity with this method is the adaptive control of 3D Gaussians during training so the representation matches the scene by dynamically adding, splitting, and removing 3D Gaussians during training. The previously mentioned $$\alpha$$ is used here to control the opacity of Gaussians and effectively measures how much a Gaussian contributes to a rendered image. When Gaussians have persistently low opacity they are pruned, which prevents wasted computation on empty space and is a key reason for the computational efficiency of this approach. On the other hand, large Gaussians are split in this step to allow for a better fit of detailed surfaces, and Gaussians with large gradients are duplicated to increase local density.

Finally, the key aspect of the method that allows for incredibly competitive training times and real-time rendering is the Fast Differentiable Rasterizer for Gaussians. This custom GPU rasterization is designed to efficiently render and differentiate through large numbers of anisotropic Gaussian splats, while avoiding the scalability and gradient-truncation issues of prior splatting methods. It is created with three main objectives: fast rendering and sorting of Gaussian splats, correct (or near-correct) alpha blending for anisotropic Gaussians, and unrestricted gradient flow. The key idea is to sort all Gaussian splats once per image, rather than per pixel, and then reuse this ordering during rasterization utilizing a tile-based pipeline. This key innovation allows for 3D Gaussian Splatting to be practical at scale and speed.

#### Architecture

As opposed to previous state-of-the-art methods such as NeRF, which utilize neural architectures built around MLPs, 3D Gaussian Splatting is an explicit geometric approach built around learnable primitives and rasterization.

The paper proposes a scene-level, non-neural architecture composed of three tightly coupled modules:

1. Explicit 3D Gaussian Scene Representation
2. Optimization and Adaptive Density Control
3. Fast Differentiable Tile-Based Rasterizer

The expressivity is moved from neural layers into the geometric primitives of 3D Gaussians as discussed above. The main training loop and optimization is shown below.

![3DGS Architecture]({{ '/assets/images/32/3dgs_architecture.png' | relative_url }})
{: style="width: 700px; max-width: 100%;"}
_Fig. Optimization of sparse SfM point cloud to create a set of 3D Gaussians, which are tiled for the optimization strategy by rasterization._

The optimization starts with a sparse SfM point cloud, which is used at initialization to produce the set of initial 3D Gaussian parameters. Each SfM point provides: mean (3D position), isotropic covariance (size), opacity $$\alpha$$, and color (via spherical harmonics).

Next, this 3D representation enters the training loop at the projection stage, where camera intrinsics and extrinsics are used as inputs along with the transform of the 3D covariance matrices to create 2D anisotropic covariance matrices. This is effectively the splatting stage of the architecture.

From here, the Differentiable Tile Rasterizer renders the image as described in the methodology above, efficiently creating the 2D pixel-level representation. The rendered image is then compared against the ground-truth training image, and the photometric loss ($$L_1$$ + SSIM) is used to compute a gradient that is passed back for SGD.

Passing through the differentiable rasterizer, the gradients flow through both the projection to 3D Gaussians for parameter updates and to the Adaptive Density Control module, which uses these gradients to dynamically adjust the number of Gaussians and the spatial resolution of the representation.

#### Results

Bernhard et al. tested the model for image synthesis against previous NeRF methods.

The paper evaluated 3D Gaussian Splatting across a diverse set of datasets against 13 real scenes from published datasets and the synthetic Blender dataset. The scenes were chosen for different capture styles, covering both indoor and outdoor environments. Results are compared to state-of-the-art quality methods at the time (Mip-NeRF360) as well as the fastest NeRF methods (InstantNGP and Plenoxels) as benchmarks for quality and speed, respectively.

![Comparison of 3DGS to previous methods]({{ '/assets/images/32/3dgs_comparison_photos.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
_Fig. Comparison of 3D Gaussian Splatting to previous methods and ground truth from the Mip-NeRF360 dataset._

The image comparisons above demonstrate that 3D Gaussian Splatting achieves comparable quality to Mip-NeRF360. Critical examples are highlighted in each image with zoom-ins or red arrows.

![Results Table]({{ '/assets/images/32/3dgs_results_table.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
_Table 1. Quantitative evaluation of 3DGS (labeled "Ours-7k" and "Ours-30k") compared to previous work, computed over three datasets. Results marked with a dagger (†) have been directly adopted from the original paper; all others were obtained in the 3DGS paper._

For a more quantitative focus, the table above shows that 3D Gaussian Splatting either performs comparably to or outperforms NeRF methods in quality measures, while achieving low training times proportional to quality outputs. A critical value to note is the FPS of 3D Gaussian Splatting, which for the first time enabled real-time rendering speeds. The main drawback, aside from comparison to Plenoxels, is that 3D Gaussian Splatting is highly memory-intensive, requiring significantly more storage than NeRF methods.

#### Discussion

3D Gaussian Splatting (3DGS) is best understood as a shift in where radiance-field expressivity lives. At the time of the paper, NeRF-style methods were largely state-of-the-art and concentrated representation power inside neural functions, typically through MLPs. 3DGS moves the representational power to a large set of explicit, learnable primitives—anisotropic Gaussians—so that rendering can become closer to typical software rasterization of computer scenes rather than volumetric ray marching, which is much more computationally intensive and less parallelizable. This effectively removes a bottleneck that had prevented previous real-time novel view synthesis at high resolution. By rendering only the primitives that project onto the screen and compositing them efficiently, 3DGS avoids wasting computation on empty space and achieves high framerates without sacrificing radiance-field-style image formation.

3DGS couples this effective representation with an optimization strategy that refines the structure of the image. Beyond strict updates of Gaussian attributes, the method also adaptively changes the representation through pruning and densification of existing Gaussians. This helps improve reconstruction while avoiding wasted computation.

The rasterizer architecture is what makes this approach feasible, as it accelerates both training and inference while preserving differentiability and stable gradient flow. As with most performance-driven designs, the tile-based pipeline trades compute for memory: the same tiling and primitive management that enable speed also increase the storage footprint, making 3DGS substantially more memory-intensive due to the large number of stored primitives.

Overall, 3DGS takes a fundamentally different approach from previous methods of radiance-field-quality novel view synthesis by storing representational power in 3D Gaussian primitives and utilizing optimizations and architecture to make it feasible. This leads to a tradeoff: substantially faster rendering with competitive visual quality, but high memory costs.

#### Running the Codebase

##### Exploring Activation Functions

<!-- PASTE: Content about activation function exploration -->

##### Clamping

<!-- PASTE: Clamping experiment content -->
<!-- NOTE: Include your code block and results here -->

---

### Large View Synthesis Model (LVSM)

#### Overview

<!-- PASTE: LVSM Overview content (starts with "Despite advances such as 3D Gaussian Splatting...") -->

#### Method

<!-- PASTE: LVSM Method content (starts with "Given N sparse input images...") -->
<!-- NOTE: You'll need to add your formulas here using LaTeX $$ syntax -->

#### Architecture

<!-- PASTE: LVSM Architecture content (starts with "The paper introduces the Large View Synthesis Model...") -->

#### Results

<!-- PASTE: LVSM Results content (starts with "Jin et. al tested the LVSM model...") -->
<!-- NOTE: Add your object-level and scene-level comparison figures here -->

#### Discussion

<!-- PASTE: LVSM Discussion content (starts with "The decoder-only model shows better performance...") -->

#### Running the Codebase

<!-- PASTE: LVSM codebase running content -->
<!-- NOTE: Include your video link and Colab link here -->

---

## Conclusion

The comparison between 3D Gaussian Splatting and the Large View Synthesis Model shows a variety of approaches in novel-view synthesis from geometry-based approaches to data-driven paradigms. 3D Gaussian Splatting demonstrates that explicit representations using Gaussians can achieve real-time performance and photorealistic quality through efficient rendering and adaptive optimization but at a high memory cost. In contrast, LVSM challenges the need for strong 3D inductive biases, achieving competitive results by leveraging transformer architectures and large-scale training for generalizable view synthesis. Together, these approaches illustrate two distinct yet complementary paths toward faster, more flexible, and increasingly general 3D scene understanding.

## Reference

[1] Kerbl, Bernhard, et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." _ACM Transactions on Graphics (SIGGRAPH)_, vol. 42, no. 4, 2023.

---
