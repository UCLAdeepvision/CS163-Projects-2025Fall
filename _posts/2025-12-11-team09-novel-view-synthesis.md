---
layout: post
comments: true
title: Novel View Synthesis
author: Vivek Alumootil and Jinying Lin
date: 2025-12-11
---

##### In computer graphics and vision, novel view synthesis is the task of generating images of a scene given a set of images of the same scene taken from different perspectives. In this report, we introduce three important papers attempting to solve this task.  

### <u>LVSM: A Large View Synthesis Model With Minimal 3D Inductive Bias</u>

Most existing novel view synthesis methods rely on explicit 3D scene representations, such as NeRF-style volumetric fields [5] or 3D Gaussian Splatting [6]. While effective, these approaches impose strong geometric inductive biases through predefined 3D structures and handcrafted rendering equations. LVSM proposes a fundamentally different approach by minimizing 3D inductive bias and reformulating novel view synthesis as a direct image-to-image prediction task conditioned on camera poses.

#### Method

##### Token-Based Representation

Instead of constructing an explicit 3D representation as in NeRF or 3DGS, LVSM operates entirely in token space using a large transformer model. Given $$N$$ input images with poses $$\{(I_i, E_i, K_i)\}_{i=1}^N$$, LVSM computes pixel-wise Plücker ray embeddings $$P_i \in \mathbb{R}^{H \times W \times 6}$$ that encode ray origin and direction in a continuous 6D parametrization. The images and Plücker embeddings are patchified and projected into tokens:

<center>
$$x_{ij} = \text{Linear}_{\text{input}}([I_{ij}, P_{ij}]) \in \mathbb{R}^d$$
</center>

Target views are similarly represented by tokenized Plücker ray embeddings $$q_j$$.

##### Encoder-Decoder Architecture

The encoder–decoder architecture introduces fixed-size learnable latent scene tokens ($$L = 3072$$) that aggregate information from all input views through bidirectional self-attention:

<center>
$$x'_1, \ldots, x'_{l_x}, z_1, \ldots, z_L = \text{Transformer}_{\text{Enc}}(x_1, \ldots, x_{l_x}, e_1, \ldots, e_L)$$
</center>

<center>
$$z'_1, \ldots, z'_L, y_1, \ldots, y_{l_q} = \text{Transformer}_{\text{Dec}}(z_1, \ldots, z_L, q_1, \ldots, q_{l_q})$$
</center>

This compact representation enables inference time independent of the number of input views.

##### Decoder-Only Architecture

The decoder-only architecture processes input and target tokens jointly in a single stream, removing the latent bottleneck:

<center>
$$x'_1, \ldots, x'_{l_x}, y_1, \ldots, y_{l_q} = \text{Transformer}_{\text{Dec-only}}(x_1, \ldots, x_{l_x}, q_1, \ldots, q_{l_q})$$
</center>

Although computationally costlier due to quadratic attention, it demonstrates superior scalability with increasing input views.

Both architectures regress RGB values using $$\hat{I}^t_j = \text{Sigmoid}(\text{Linear}_{\text{out}}(y_j))$$ and are trained end-to-end with MSE and perceptual loss, without depth supervision or geometric constraints. QK-Normalization stabilizes training.

#### Results

LVSM achieves state-of-the-art performance on object-level (ABO, GSO) and scene-level (RealEstate10K) datasets, improving PSNR by 1.5–3.5 dB over GS-LRM [2], particularly excelling on specular materials, thin structures, and fine textures. The decoder-only model shows strong zero-shot generalization: trained with four views, it continues improving up to sixteen views. Conversely, the encoder-decoder model plateaus beyond eight views due to its fixed latent representation. Notably, even small LVSM models trained on limited resources outperform prior methods.

#### Discussion

LVSM demonstrates that high-quality novel view synthesis can be achieved through purely data-driven learning without explicit 3D representations. However, as a deterministic model, it struggles to hallucinate unseen regions, producing noisy artifacts rather than smooth extrapolation. The decoder-only architecture also becomes expensive with many input views.

The superior performance of decoder-only over encoder-decoder reveals an important insight: 3D scenes may resist compression into fixed-length representations. Unlike language, visual information from multiple viewpoints contains redundant spatial structure that degrades when forced through a latent bottleneck, explaining why encoder-decoder performance drops beyond eight views while decoder-only continues scaling.

Overall, LVSM demonstrates that learned representations without structural constraints offer a promising direction for novel view synthesis, echoing the shift from encoder–decoder to decoder-only architectures in large language models.

### <u>RayZer: A Self-supervised Large View Synthesis Model</u>

The dominant paradigm in studies on novel view synthesis has been to train models on datasts with COLMAP [1] annotations. However, there are several notable drawbacks to this approach. COLMAP is slow, struggles in sparse-view and dynamic scenarios and may produce noisy annotations. RayZer introduces a groundbreaking new approach to novel view synthesis that is entirely self-supervised (e.g. it does not use any camera pose annotations during training or testing) and thus avoids COLMAP entirely.

Along with SRT [2] and LVSM, RayZer falls into the group of novel view synthesis methods employing a purely learned latent scene representation and a neural network to render it. This is in contrast to most state-of-the-art methods, which primarily rely on NeRF [3] or 3D Gaussian [4] Representations. These two techniques carry a strong inductive bias through their use of volumetric rendering.
 
#### Method

##### Self-Supervised Learning
RayZer is a self-supervised method that takes in a set of unposed multiview images and outputs geometric and positional information (e.g camera poses and camera intrinsics), as well as the latent scene representation. The input images are split into two subsets $$I_A$$ and $$I_B$$. $$I_A$$ is used to predict the scene representation, and $$I_B$$ supervises this prediction by producing a loss between the predicted renders of the scene representation from $$I_B$$ and the ground truth. 

<div style="text-align: center;"> 
![YOLO]({{ '/assets/images/9/RAYZER.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. The RayZer framework consists of three stages: Camera Estimation, Latent Scene Reconstruction and Rendering
</div>

##### Camera Estimation
RayZer first patchifies the image and converts the patches into tokens with a linear layer. Both intra-frame spatial positional encodings and inter-frame image index positional encodings are employed. To predict the pose of each camera (with respect to a chosen canonical view), RayZer allows information to flow from the image tokens to camera tokens, one for each image, by integrating both of them into a transformer, and then decodes them with an MLP. Specifically, the camera tokens are initialized with a learnable value $$\mathbb{R}^{1 \times d}$$. An image index positional encoding is added to them to inform the model which image each camera token corresponds to. The camera estimator consists of several full self-attention layers. Each layer can be written like so: 
<center>
$$
y^0 = \{f, p\}
$$
$$
y^l = \text{TransformerLayer}^l(y^{l-1})
$$
$$
\{f*, p* \} = \text{split}(y^{l_T}) 
$$
</center>
The first equation describes how the initial $$y$$ value, $$y^{0}$$, is initialized as the concatenation of $$f$$, the extracted image features, and $$p$$, the intial camera tokens. The second equation describes the how $y$ is repeatedly updated by passing it through a transformer layer. The third equation describes how $$f^{*}$$ and $$p^{*}$$ are extracted from the final $$y$$ value.

Finally, the relative pose for each image is estimated with a MLP like so: 
<center>
$$p_i= \text{MLP}_{\text{Pose}}([p_i^{*}, p_c^{*}]),$$
</center>
where $$p_i$$ is the camera token for image $$I_i$$ and $$p_c$$ is the camera token for the canonical view. The output is parametrized with a continuous 6D representation. Another similar MLP is used to estimate the focal length from the camera token corresponding to the canonical view.

##### Scene Reconstructor

We can generate a Plucker Ray map [5], a common technique to represent cameras, for each image in $$A$$ by combining the predicted SE(3) pose $$P_i$$ and intrinsic matrix $$K$$ (which is derived from the predicted focal length). The Plucker ray maps are patchified and then combined with the image tokens of $$A$$ along the feature dimension using an MLP. The result of this fusing process is $$x_A$$. 

The scene reconstructor uses an identical mechanism to the camera estimation module. Learnable tokens for the scene reconstruction $$z$$, with $$z \in \mathbb{R}^{L \times d}$$ for a latent scene of $$L$$ tokens, are updated through several transformer layers along with $$x_A$$. $$x_A$$ is discarded at the end of the sequence and the the final value of $$z$$, $$z^{*}$$, is the latent scene representation.

##### Rendering Decoder

The goal of the rendering decoder is to render the scene from novel viewpoints using the latent scene representation. The architecture is inspired by LVSM. To render a target camera, we represent the target camera as a Plucker ray map, tokenize it and use the update rule
<center>
$${r^{*}, z'} = D_{\text{render}}(\{r, z^{*}\})$$
</center>   
where $$r^{*}$$ are the tokens that are used for rendering. $$D_{\text{render}}$$ uses the same Transformer layer mechanism as the camera estimation and scene reconstructor modules. Finally, to produce the final RGB image, we decode the render tokens with an MLP like so:
<center>
$$\hat{I} = \text{MLP}_{\text{rgb}}(r^{*})$$
</center>

##### Results

Despite not using any ground truth camera annotations during training or testing, RayZer achieves state of the art performance. It achieves a 2.82% better PSNR, 4.70% better SSIM and 13.6% better LPIPS on the DL3DV-10K dataset [6] compared to LVSM, a leading method in novel view synthesis. The paper highlights how this may be in part due to the noisy nature of COLMAP annotations; by learning from the raw data itself, RayZer may have a higher ceiling than methods supervised by inaccurate camera pose data.

  
##### Discussion

Perhaps the most compelling and fascinating part about RayZer is that it is able to estimate camera poses without ever being shown ground truth camera pose data. Neural networks tend to work with high-dimensional, abstract features, and it is uncommon to see them estimate simple, understandable values withotu supervision. RayZer is in this case is guided by the dimension of the output; if the 6 DoF parametrization was never forced to be a bottleneck of information flow in the network, it would likely never adopt such a simple representation. This technique of forcing a neural network to conver its abstract, latent representation to an useful, understandable value by constraining its dimensions, almost like an autoencoder, is promising.  

### References

[1] Schonberger, Johannes L., and Jan-Michael Frahm. "Structure-from-motion revisited." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[2] Sajjadi, Mehdi SM, et al. "Scene representation transformer: Geometry-free novel view synthesis through set-latent scene representations." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[3] Mildenhall, Ben, et al. "Nerf: Representing scenes as neural radiance fields for view synthesis." Communications of the ACM 65.1 (2021): 99-106.

[4] Kerbl, Bernhard, et al. "3D Gaussian splatting for real-time radiance field rendering." ACM Trans. Graph. 42.4 (2023): 139-1.

[5] Zhang, Jason Y., et al. "Cameras as rays: Pose estimation via ray diffusion." arXiv preprint arXiv:2402.14817 (2024).

[6] Ling, Lu, et al. "Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
