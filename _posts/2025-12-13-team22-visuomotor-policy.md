---
layout: post
comments: true
title: Visuomotor Policy
author: Haoran Li
date: 2025-12-13
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

## Problem Formulation

## Approaches

## Policy Pretraining

Learning visuomotor policies directly from interaction data is often prohibitively expensive, particularly in real-world robotic and autonomous driving settings. As a result, many approaches seek to improve sample efficiency by pretraining visual representations or policy components prior to downstream imitation or reinforcement learning. However, existing methods differ substantially in both how pretraining is performed and whether it consistently benefits policy learning.

In this section, we examine two complementary perspectives on policy pretraining. The first presents an action-conditioned contrastive learning framework that explicitly aligns representation learning with control objectives [1]. The second revisits the effectiveness of visual pretraining by comparing it against strong learning-from-scratch baselines, providing a critical assessment of when and why pretraining improves visuomotor policy learning [2].

### Action-Conditioned Contrastive Policy Pretraining

In *Learning to Drive by Watching YouTube Videos: Action-Conditioned Contrastive Policy Pretraining* [1], Zhang et al. introduce Action-Conditioned Contrastive Policy Pretraining (ACO), a contrastive representation learning framework designed specifically for visuomotor policy learning. The work builds on MoCo-style self-supervised learning, but argues that standard visual contrastive objectives are insufficient for control tasks because they primarily encode appearance-based features rather than action-relevant semantics. ACO modifies the contrastive objective to explicitly incorporate action information, enabling representations that are better aligned with downstream imitation and reinforcement learning.

#### Motivation

Traditional visual pretraining methods, such as ImageNet supervision or instance-level contrastive learning, aim to distinguish images based on visual content. However, in visuomotor control, the key requirement is not visual discrimination per se, but the ability to identify which aspects of an observation are predictive of the correct action. For example, road curvature and lane markings are critical for driving decisions, whereas lighting conditions or background buildings are largely irrelevant. The authors hypothesize that contrastive pretraining should group observations that require similar actions, even if they differ significantly in visual appearance.

#### Method Overview

ACO extends standard contrastive learning by combining two types of positive pairs:

* Instance Contrastive Pairs (ICP): as in MoCo, two augmented views of the same image form a positive pair, encouraging general visual discriminability.
* Action Contrastive Pairs (ACP): two different images are treated as a positive pair if their associated actions are sufficiently similar (e.g., steering angles within a predefined threshold), regardless of visual similarity.

By jointly optimizing ICP and ACP objectives, ACO encourages representations that preserve general visual structure while emphasizing action-consistent features. This dual-objective design allows the model to retain the benefits of instance discrimination while mitigating its tendency to overfit to appearance-level cues.

#### Pseudo Action Label Generation

A central challenge in applying ACP at scale is the lack of action annotations in in-the-wild data such as YouTube videos. To address this, Zhang et al. train an inverse dynamics model on the NuScenes dataset, where ground-truth ego-motion actions are available. Given consecutive frames $$ (o_t, o_{t+1}) $$, the inverse dynamics model predicts the corresponding control action

$$
\hat{a}*t = h(o_t, o*{t+1}),
$$

where $$ h $$ is trained using supervised regression. The trained inverse dynamics model is then applied to unlabeled YouTube driving videos to generate pseudo action labels, enabling large-scale action-conditioned pretraining without manual annotation. Importantly, optical flow is used as input to the inverse dynamics model instead of raw RGB frames, improving robustness to appearance variation and reducing prediction error.

![Inverse Dynamics Pipeline]({{ '/assets/images/team22/aco_inverse_dynamics.png' | relative_url }})
{: style="width: 500px; max-width: 100%;"}
*Fig. 1. Inverse dynamics training on NuScenes and pseudo action label generation on YouTube driving videos [1].*

#### Training Architecture and Objective

The ACO architecture follows the MoCo framework and consists of a query encoder $$ f $$ and a momentum-updated key encoder $$ f_m $$. A shared visual backbone feeds into two projection heads: $$ g_{\text{ins}} $$ for the ICP objective and $$ g_{\text{act}} $$ for the ACP objective. Momentum-smoothed counterparts $$ g_{\text{ins},m} $$ and $$ g_{\text{act},m} $$ are used to populate a dictionary of negative samples.

The ICP loss follows the standard InfoNCE formulation:

$$
\mathcal{L}_{\text{ins}} =

* \log
  \frac{\exp(z^q \cdot z^+ / \tau)}
  {\sum_{z \in \mathcal{N}(z^q)} \exp(z^q \cdot z / \tau)},
  $$

where $$ z^q $$ is the query embedding, $$ z^+ $$ is the positive key, $$ \mathcal{N}(z^q) $$ is the set of negatives, and $$ \tau $$ is a temperature parameter.

For ACP, the positive set is defined based on action similarity:

$$
\mathcal{P}_{\text{act}}(z^q) =

{ z \mid \lVert \hat{a} - \hat{a}^q \rVert < \epsilon,\ (z,\hat{a}) \in \mathcal{K} },
$$

where $$ \epsilon $$ is a predefined action-distance threshold and $$ \mathcal{K} $$ denotes the key set. The ACP loss is then given by

$$
\mathcal{L}_{\text{act}} =

* \log
  \frac{\sum_{z^+ \in \mathcal{P}*{\text{act}}(z^q)} \exp(z^q \cdot z^+ / \tau)}
  {\sum*{z \in \mathcal{N}(z^q)} \exp(z^q \cdot z / \tau)}.
  $$

The final training objective is a weighted combination of the two losses:

$$
\mathcal{L} =

\lambda_{\text{ins}} \mathcal{L}*{\text{ins}}
+
\lambda*{\text{act}} \mathcal{L}_{\text{act}}.
$$

![ACO Architecture]({{ '/assets/images/team22/aco_architecture.png' | relative_url }})
{: style="width: 500px; max-width: 100%;"}
*Fig. 2. ACO architecture showing joint optimization of instance contrastive and action-conditioned contrastive objectives [1].*

#### Experimental Results

The authors evaluate ACO on downstream imitation learning tasks in the CARLA driving simulator, comparing against random initialization, autoencoder pretraining, ImageNet pretraining, and MoCo. Across all demonstration sizes, ACO consistently outperforms the baselines, with the performance gap most pronounced in low-data regimes. These results indicate that action-conditioned pretraining substantially improves sample efficiency and leads to representations that are better aligned with downstream control objectives.

![ACO Imitation Learning Results]({{ '/assets/images/team22/aco_results.png' | relative_url }})
{: style="width: 500px; max-width: 100%;"}
*Fig. 3. Imitation learning performance under varying demonstration sizes for different pretraining methods [1].*

### Pretraining Versus Learning-from-Scratch

In *On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline* [2], Hansen et al. provide a systematic re-evaluation of visual policy pretraining by comparing frozen pretrained representations against strong learning-from-scratch baselines. Rather than proposing a new pretraining method, this work examines whether commonly used pretrained visual encoders offer consistent advantages once baselines are properly controlled.

#### Problem Formulation

Learning-from-scratch refers to training a randomly initialized encoder $$ f_\theta $$ jointly with a policy head using only task-specific data:

$$
\theta^* = \arg\min_\theta ; \mathbb{E}*{(o,a)\sim\mathcal{D}} \left[ \ell(\pi*\theta(o), a) \right],
$$

where $$ \ell $$ denotes a behavior cloning or reinforcement learning objective.

In contrast, frozen pretraining fixes a pretrained encoder $$ f_{\text{pre}} $$ and optimizes only the policy parameters:

$$
\theta^* = \arg\min_\theta ; \mathbb{E}*{(o,a)\sim\mathcal{D}} \left[ \ell(\pi*\theta(f_{\text{pre}}(o)), a) \right].
$$

The authors emphasize that many prior works compare pretrained models against weak learning-from-scratch baselines that lack sufficient capacity or data augmentation.

#### Experimental Setup and Findings

Hansen et al. evaluate learning-from-scratch and frozen pretraining across multiple domains, including Adroit, DMControl, PixMC, and real-world robot manipulation, using behavior cloning, PPO, and DrQ-v2. The learning-from-scratch baselines employ shallow convolutional encoders paired with strong data augmentation, such as random shift and color jitter:

$$
o' \sim \mathcal{T}(o),
$$

which substantially improves robustness and generalization.

![LfS vs Pretraining Overview]({{ '/assets/images/team22/lfs_vs_pretraining.png' | relative_url }})
{: style="width: 500px; max-width: 100%;"}
*Fig. 4. Comparison of learning-from-scratch and frozen pretrained representations across multiple domains [2].*

Across most tasks and learning algorithms, learning-from-scratch with augmentation matches or exceeds the performance of frozen pretrained representations. Frozen pretraining provides only limited benefits in very low-data regimes, and no single pretrained representation consistently dominates across domains. The authors attribute this in part to a domain gap between pretraining data (real-world images and videos) and downstream environments (often simulated).

The authors further show that finetuning pretrained encoders on in-domain data,

$$
\theta^* = \arg\min_\theta ; \mathbb{E}*{(o,a)\sim\mathcal{D}} \left[ \ell(\pi*\theta(f_\theta(o)), a) \right],
$$

can outperform both frozen pretraining and learning-from-scratch when combined with strong augmentation, highlighting the importance of domain adaptation.

### Summary and Discussion

Taken together, the two works reviewed in this section offer complementary perspectives on policy pretraining for visuomotor control. Action-Conditioned Contrastive Policy Pretraining [1] demonstrates that pretraining can substantially improve downstream policy learning when the objective is explicitly aligned with action semantics, rather than relying solely on appearance-based visual similarity. By incorporating action-conditioned positives and leveraging large-scale unlabeled video data, ACO learns representations that are more directly relevant to control, yielding significant gains in low-data imitation and reinforcement learning settings.

In contrast, Hansen et al. [2] provide a critical re-evaluation of visual policy pretraining, showing that frozen pretrained representations do not consistently outperform strong learning-from-scratch baselines when data augmentation and architectural choices are properly controlled. Their findings highlight the importance of domain alignment and adaptation, suggesting that generic visual pretraining alone is insufficient to guarantee improvements in visuomotor policy learning.

Together, these results suggest that the effectiveness of policy pretraining depends not only on dataset scale, but more critically on the alignment between the pretraining objective and downstream control tasks. While action-aware and task-aligned pretraining objectives show clear promise, future work must carefully consider domain gaps, finetuning strategies, and strong baseline comparisons to fully realize the benefits of pretraining in visuomotor learning.

## References

[1] Zhang, Qihang, et al. “Learning to Drive by Watching YouTube Videos: Action-Conditioned Contrastive Policy Pretraining.” European Conference on Computer Vision, 2022.

[2] Hansen, Nicklas, et al. “On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline.” International Conference on Learning Representations, 2023.
