---
layout: post
comments: true
title: Robot Navigation Using Deep Vision Models
author: Ryan Teoh, Bill, Maddox, Andrew
date: 2025-12-10
---

> In this project we build a full robot navigation pipeline inside an embodied AI simulation (AI2-THOR).  
> Our system detects objects, understands spatial relationships, and navigates toward a target object using modern vision models such as **CLIP**, **SAM (Segment Anything)**, and multimodal reasoning modules.  
> We demonstrate an end-to-end system capable of object identification, object-centric navigation, and spatial reasoning.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Overview

Embodied AI requires agents to perceive, reason, and act in realistic 3D environments.  
We use **AI2-THOR**, an interactive simulation containing kitchens, living rooms, bedrooms, and bathrooms.  
Our goal is to enable an agent to:

1. Identify target objects using vision-language models (CLIP)
2. Segment the target and surrounding context using SAM
3. Understand spatial relationships (e.g., *"the laptop is on the sofa"*)
4. Move toward the target using closed-loop visual feedback

To achieve this, we integrate **SAM**, **CLIP**, and a custom spatial-reasoning module into a perception–action pipeline.

---

## Computer Vision Model Architecture

### 1. **Object Identification with CLIP**

We use CLIP to match image patches against text prompts for object names.

![CLIP Pipeline]({{ '/assets/images/team-id/clip_pipeline.png' | relative_url }})
{: style="width: 500px; max-width: 100%;"}
*Fig 1. CLIP identifies the target object using text–image similarity.*

Given a target command (e.g., *"find the microwave"*), the system:

1. Samples candidate bounding boxes
2. Extracts embeddings via CLIP image encoder
3. Computes similarity with the text embedding
4. Selects the highest-scoring region as the target

---

### 2. **Segmentation with SAM (Segment Anything)**

After locating the target region, we refine the mask using SAM:

![SAM Example]({{ '/assets/images/team-id/sam_mask.png' | relative_url }})
{: style="width: 500px;"}

SAM provides a high-quality segmentation mask, which we use for:

- Object localization
- Pixel-level spatial reasoning
- Deriving the agent’s navigation targets

---

### 3. **Spatial Relationship Detection**

We extend the system to detect relations like:

- *"X is on Y"*
- *"X is next to Y"*
- *"X is inside Y"*

Using metadata from AI2-THOR combined with SAM masks:

```
# This is a sample code block
def get_spatial_context(controller, target_mask):
    """
    Uses AI2-THOR metadata + instance segmentation
    to determine which object the target is on or inside.
    """
```

## Navigating the Enviornment

## Example Demonstration

## Conclusions

## Code

Project Repo: [GitHub Repository](https://github.com/Land-dev/finalProject163)

SAM repo:

CLIP repo:

Ai2-Thor simulation (We use RoboThor): [RoboThor](https://ai2thor.allenai.org/robothor)

## References
