---
layout: post
comments: true
title: Camera Pose Estimation
author: Group 28
date: 2025-12-07
---

> Camera pose estimation is a fundamental Computer Vision task that aims to determine the position and orientation of a camera relative to a scene with image/video data. Our project evaluates different pose estimation methods (COLMAP, VGGSfM, and ViPE) on video datasets using quantitative performance metrics. 

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Camera pose estimation has become a fundamental task in Computer Vision 
that aims to determine the position (translation) and orientation (rotation) of a camera relative to a scene using image or extracted video data. Accurately estimating the absolute pose of a camera has widespread applications in 3D reconstruction, world models, and augmented reality.

In this project, we are evaluating three camera pose estimation methods: COLMAP, VGGSfM, and ViPE on the same dataset with a set of quantitative metrics. We describe each approach and analyze their relative strengths and limitations. 

### Camera Pose Estimation
For camera pose estimation, there are 3D-2D correspondences between a 3D point in the world (scene geometry) and the 2D pixel location where that point appears in the image or video frame. 

Camera pose estimation predicts the pose of a camera with these two components:
- A translation vector: which describes where the camera is in the world coordinate system
- A rotation: which describes the camera's orientation relative to the world

![YOLO]({{ '/assets/images/team28/pose_estimation.PNG' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Pose estimation formula* [1].

### COLMAP
COLMAP is an end-to-end 3D reconstruction pipeline that estimates both scene geometry and camer poses from images. It uses Structure-from-Motion (SfM) to recover a sparse representation of the scene and camera poses of the input images. This is then fed into Multi-View Stereo (MVS) which recovers a dense representation of the scene. 

The process of SfM consists of taking a series of images and then performing feature detection and extraction, feature matching and geometric verification, and structure and motion reconstruction. 

For the feature matching and geometric verification, we used sequential matching which is best for images that were acquired in sequential order, like by a video camera. Since frames have visual overlap, it is not required to use exhaustive matching. In this process, consecutive images and matched against each other. 

Multi-View Stereo (MVS) then takes what was output from SfM to compute depth and/or normal information of every pixel in the image, and then it uses the depth and normal maps to create a dense point cloud of the scene. This sparse reconstruction process loads the extracted data from the database and incrementally extends the reconstruction from an initial image pair by registering new image and triangulating new points. 



https://demuc.de/colmap/
https://colmap.github.io/tutorial.html 


<div style="display: flex; gap: 20px; justify-content: center;">
  <div style="text-align: center;">
    <img src="{{ '/assets/images/team28/colmap_demo.gif' | relative_url }}"
         style="width: 400px; max-width: 100%;">
    <p><em>(a) COLMAP sparse reconstruction on freiburg1_plant</em></p>
  </div>

  <div style="text-align: center;">
    <img src="{{ '/assets/images/team28/colmap_demo2.gif' | relative_url }}"
         style="width: 400px; max-width: 100%;">
    <p><em>(b) Alternate view of camera trajectory and sparse points</em></p>
  </div>
</div>
<p style="text-align: center;">
<em>Fig. 2. COLMAP sparse 3D points and estimated camera poses (red frustums) on the freiburg1_plant sequence.</em> [2].
</p>




<!--
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)
-->

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

[1] https://www.cs.cmu.edu/~16385/s17/Slides/11.3_Pose_Estimation.pdf

---
