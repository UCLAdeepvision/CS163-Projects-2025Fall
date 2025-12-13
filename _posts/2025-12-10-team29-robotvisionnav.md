---
layout: post
comments: true
title: "Robot Navigation Using Deep Vision Models - Project Track: Project 6"
Project Track: Project 6
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

**Background:** CLIP (Contrastive Language–Image Pretraining) is a vision–language model trained to align images and text in a shared embedding space using large-scale contrastive learning. During training, CLIP learns two encoders—a visual encoder and a text encoder—that map images and natural language descriptions into a common latent space. The model is trained on millions of image–caption pairs by maximizing the similarity between matched image–text pairs while minimizing similarity between mismatched pairs. As a result, CLIP learns rich semantic representations that capture high-level visual concepts and their corresponding linguistic descriptions without relying on fixed class labels.

We use CLIP for object identification because embodied environments contain a large variety of object instances, appearances, and viewpoints that are impractical to enumerate with fixed category classifiers. CLIP’s joint vision–language embedding space enables open-vocabulary recognition, allowing the agent to generalize to unseen objects, synonyms, and varied visual contexts without retraining.

![CLIP Pipeline]({{ '/assets/images/29/CLIP.png' | relative_url }})
{: style="width: 1000px; max-width: 100%;"}
*Fig 1. CLIP identifies the target object using text–image similarity. (Image Credit: [CLIP](https://github.com/openai/CLIP))*

Given a target command (e.g., *"find the microwave"*), the system:

1. Samples candidate bounding boxes
2. Extracts embeddings via CLIP image encoder
3. Computes similarity with the text embedding
4. Selects the highest-scoring region as the target

CLIP allows us to compute the semantic similarity between:

- a text embedding (ex. cup)
- an image embedding (ex. cropped SAM mask region)

In our pipeline, SAM and CLIP serve complementary roles. SAM is responsible for proposing high-quality, class-agnostic object masks, while CLIP assigns semantic meaning to each mask by scoring its visual content against the language query. This separation allows precise spatial localization without sacrificing semantic flexibility.

For each candidate object mask $$m_i$$ produced by SAM, we compute a semantic similarity score with the language query 
$$q$$ using CLIP’s joint vision–language embedding space:

$$
s_i =
\frac{
f_{\text{img}}(m_i)^\top f_{\text{text}}(q)
}{
\left\lVert f_{\text{img}}(m_i) \right\rVert_2
\left\lVert f_{\text{text}}(q) \right\rVert_2
}
$$

where $$f_{\text{img}}(⋅)$$ and $$f_{\text{text}}(⋅)$$ denote the CLIP image and text encoders, respectively. The mask with the highest cosine similarity score is selected as the target object. We apply a softmax over similarity scores to normalize confidence across candidate masks and select the most likely target region. To improve robustness, CLIP similarity is computed on cropped SAM mask regions rather than the full image. This reduces background bias and ensures that similarity scores are dominated by object appearance rather than surrounding scene context, which is especially important in cluttered indoor environments.

In `find_best_object` we evaluate our SAM mask crop against the test query.

```
text_tokens = tokenizer([clean_query]).to(device)
image_inputs = torch.stack([clip_preprocess(img) for img in crops]).to(device)

with torch.no_grad():
    img_features = clip_model.encode_image(image_inputs)
    txt_features = clip_model.encode_text(text_tokens)

    img_features /= img_features.norm(dim=-1, keepdim=True)
    txt_features /= txt_features.norm(dim=-1, keepdim=True)

    # Softmax over similarity scores
    probs = (100.0 * img_features @ txt_features.T).softmax(dim=0)
    values, indices = probs.topk(1)
```

By cleaning the text query and computing similarity scores, we can pick the highest scoring object mask to obtain our predicted object for the query. We use our best mask and score to draw our bounding boxes later as well as perform spatial reasoning and navigation. CLIP is also beneficial for our model because objects may be partially hidden, rotated, etc. in our robot simulation, and SAM alongside CLIP can determine whether a segment looks like the queried box. While CLIP provides strong semantic generalization, performance may degrade under severe occlusion, extreme viewpoint distortion, or visually ambiguous objects. To address these cases, our system leverages closed-loop navigation, allowing the agent to reposition and re-evaluate candidate masks from improved viewpoints.


### 2. **Segmentation with SAM (Segment Anything)**

**Background:** The Segment Anything Model (SAM) is a foundation model for image segmentation designed to produce high-quality, class-agnostic masks for arbitrary objects in an image. SAM consists of a large vision encoder and a lightweight mask decoder that can generate segmentation masks from flexible prompts such as points, bounding boxes, or coarse region proposals. Trained on a diverse dataset of images and masks, SAM generalizes across object categories and visual domains without task-specific retraining, enabling robust instance segmentation even for previously unseen objects. This makes SAM particularly well-suited for embodied environments, where object appearance, scale, and occlusion vary significantly across scenes.

After locating the target region, we refine the mask using SAM:

![SAM Example]({{ '/assets/images/29/sam_mask.png' | relative_url }})
{: style="width: 1000px; max-width: 100%;"}
*Fig 2. SAM generates class-agnostic instance segmentation masks from raw RGB frames, providing candidate object regions for CLIP-based semantic matching. (Image Credit: [Segment Anything](https://github.com/facebookresearch/segment-anything))*

SAM provides a high-quality segmentation mask, which we use for:

- Object localization
- Pixel-level spatial reasoning
- Deriving the agent’s navigation targets

To obtain object candidates in each frame, we use SAM as our instance segmengation module. SAM takes in a raw RGB frame from the RoboTHOR camera and then returns a set of masks that lets us

- Object localization
- Pixel-level spatial reasoning
- Deriving the agent’s navigation targets

We use a ViT-H variant of SAM alongside a `SamAutomaticMaskGenerator`.

```
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
    pred_iou_thresh=0.3,
    stability_score_thresh=0.3,
    box_nms_thresh=0.7,
    min_mask_region_area=400,
)
```

We have a `find_best_object` helper function which performs our core perceptron segmentation to cropping to CLIP scoring.

```
masks = mask_generator.generate(image_np)

for mask_data in masks:
    x, y, w, h = map(int, mask_data["bbox"])

    if w < 10 or h < 10:
        continue

    crop = Image.fromarray(image_np[y:y+h, x:x+w])
    crops.append(crop)
    valid_masks.append(mask_data)
```

| <img src="{{ '/assets/images/29/sam_demonstration.png' | relative_url }}" alt="SAM Demonstration 1" style="width: 100%; max-width: 500px; height: auto;"> | <img src="{{ '/assets/images/29/sam_demonstration_2.png' | relative_url }}" alt="SAM Demonstration 2" style="width: 100%; max-width: 500px; height: auto;"> |
|:---:|:---:|
| *Fig 3a. Raw RGB frame captured from the robot's camera in an AI2-THOR office environment.* | *Fig 3b. Instance segmentation masks generated by SAM, with each object region highlighted in a distinct color.* |

*Fig 3. Comparison of raw camera input and SAM segmentation output.*

Figure 3 illustrates the role of SAM in converting raw visual observations into structured object-level representations. As shown in Fig. 3a, the robot’s onboard RGB camera captures a cluttered indoor scene containing multiple objects with varying scales, occlusions, and lighting conditions. Fig. 3b demonstrates SAM’s ability to generate dense, class-agnostic instance segmentation masks, where each distinct object region is separated and highlighted. These masks provide precise spatial boundaries for candidate objects, enabling downstream modules—such as CLIP-based semantic matching—to operate on isolated object regions rather than the full image. By decoupling spatial localization from semantic labeling, SAM allows the system to robustly handle novel objects and complex scenes without relying on predefined category labels.

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

Beyond simply finding what object matches the query, we also want to know where it is in the scene and whether it is on another object or on something else. Thus, to do so, we combine

1. SAM's pixel level mask for the target object.
2. Ai2-THOR's instance segmentation, telling us which object each pixel belongs to.

We thus implement a helper function called `get_spatial_context(controller, target_mask)` that allows us to

1. Read the latest event from the controller
2. Extract pixels inside the SAM mask
3. Map RGB colors to a Thor object ID
4. Lookup the object metadata and parentReceptacles. 

Intuitiviely, the SAM mask tells us which pixels are part of the target, the instance segmentation frame tells us which simulator object those pixels correspond to, and the metadata tells us what the object is residing on, such as a table, couch, drawer, etc. We can then use this location string to visualize detections, indicating a bounding box with a CLIP score and spatial context. 

WE MIGHT WANT TO INCLUDE A PICTURE OF IT FROM OUR COLLAB NOTEBOOK.

## Navigating the Enviornment

By providing the system with a natural language query, such as "find the blue cup", our model will decipher this message into a query that determines searching for a cup, in which afterwards RoboTHOR will explore the scene until the target is visible, in which we then can run our SAM / sliding window CLIP to find the object. We have two methods of navigation, one which relies on an existing API and another which uses a heuristic to calculate distance. 

GetShortestPath.

1. By using [GetShortestPath](https://ai2thor.allenai.org/robothor/documentation/#sub-shortest-path) with the AI2-THOR, we can feed parameters of the object type we are looking for, such as cup, our current position, and the allowed error. If the object exists and is achievable, then we can get the path event from the start to the end object. One limitation of our API is that it is unable to distinguish between two objects, thus if a query supplies "apple" and there exist two apples on the simulation, it will naturally navigate to the closest one. GetShortestPath provides us with the corner paths on the path, so by setting them as waypoints, our robot can follow the action until it reaches the desired object. 

```
from ai2thor.util.metrics import (
    get_shortest_path_to_object_type
)

path = get_shortest_path_to_object_type(
   controller=controller,
    object_type="Apple",
    initial_position=dict(
        x=0,
        y=0.9,
        z=0.25
    )
)
```

If our navigation does not work, then we perform random navigation, in which we consider four different directions (left, right, forward, backwards), and essentially like a maze, follow the right side of the wall until we reach the desired object. We may also perform a full rotation, and if it fails to reveal the object, it can move until it finds a new vantage point.

If the object does not exist, and if repeated movement patterns fail to detect the object, then the system concludes that the object is not in the scene and will report failure. This failsure ensures that our algorithm doesn't infinitely search forever.

## Example Demonstration

## Conclusions

In this project, we presented an end-to-end robot navigation system that integrates modern foundation vision models with embodied AI simulation to enable object-centric perception, reasoning, and navigation. By combining SAM for class-agnostic instance segmentation and CLIP for open-vocabulary semantic matching, our system is able to robustly identify target objects in cluttered, realistic indoor environments without relying on predefined object categories or task-specific retraining.

A key strength of our approach lies in the decoupling of spatial localization and semantic understanding. SAM provides precise pixel-level object boundaries, while CLIP assigns semantic meaning to each candidate region using natural language queries. This modular design allows the agent to generalize to unseen objects, handle partial occlusions, and adapt to diverse viewpoints—challenges that are common in embodied settings but difficult for traditional closed-set perception systems. Furthermore, by computing CLIP similarity over cropped segmentation masks rather than full images, we reduce background bias and improve robustness in visually complex scenes.

Beyond object detection, we extend perception to spatial relationship reasoning by leveraging AI2-THOR metadata in conjunction with SAM masks. This enables the agent to infer relations such as whether an object is on, inside, or supported by another object, allowing richer scene understanding beyond simple target localization. These spatial cues can be visualized and used to inform navigation decisions, bridging perception and action in a meaningful way.

Our navigation strategy combines simulator-provided shortest-path planning with heuristic exploration and closed-loop perception. This allows the agent to actively reposition itself when initial detections are ambiguous or incomplete, demonstrating the importance of embodied interaction for reliable visual understanding. When the target object is not present, the system terminates gracefully rather than searching indefinitely, ensuring practical behavior in failure cases.

Overall, this project demonstrates how foundation models like CLIP and SAM can be effectively integrated into an embodied AI pipeline to enable flexible, generalizable robot navigation driven by natural language commands. Future work could explore tighter integration between perception and control, multi-step language instructions, memory-based exploration, or learning-based navigation policies that further exploit the rich semantic and spatial representations provided by these models.

## Code

Project Repo: [GitHub Repository](https://github.com/Land-dev/finalProject163)

SAM repo:

CLIP repo: [Open CLIP](https://github.com/mlfoundations/open_clip)

Ai2-Thor simulation (We use RoboThor): [RoboThor](https://ai2thor.allenai.org/robothor)

## References

Ai2-Thor Documentation: https://ai2thor.allenai.org/robothor/documentation/