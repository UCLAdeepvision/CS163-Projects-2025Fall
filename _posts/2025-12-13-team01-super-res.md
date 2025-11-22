---
layout: post
comments: true
title: super res title
author: Thomas Peeler, Dylan Truong, Asher Christian, Daniel Chvat
date: 2025-01-01
---


> Abstract: survery of 3 super res models, modification of one of them; 3 models highlight diff ways of doing super res

<!--more-->
<!--bundle exec jekyll serve-->
{: class="table-of-content"}
* TOC
{:toc}

## Background and Introduction

Super-resolution is a natural problem for image-to-image methods in computer vision, referring to the process of using a degraded, downsampled image to recover the original image before degradation and downsampling. Of course, this is an ill-posed problem; two different high-resolution images can downsample into the same low-resolution image (that is, downsampling is not injective), so truly inverting the downsampling process is impossible. For this reason, we settle for the task of estimating a function that approximates an inverse, producing a high-resolution image based solely on some mathematical assumptions and the information of the low-resolution image. This includes simple, classical methods, such as nearest-neighbor or bicubic upsampling, as well as, in recent years, statistical methods making use of deep learning for computer vision.

We will be surveying three of these recent methods, highlighting the very different ways that one can go about achieving the same end goal in super-resolution:

- The Hybrid Attention Transformer (HAT), a shifted-window transformer-based method

- Look-Up Table (LUT) methods, which use precomputed table of pixel estimates

- Unfolding networks, which combine classical model-based methods with learning-based methods

Additionally, we will be discussing an experiment that we carried out involving a modification to the HAT architecture.

## A Survey of Deep Learning for Super Resolution

### Hybrid Attention Transformer (HAT)

The Hybrid Attention Transformer aims to advance the field of low-level super-resolution by improving transformer architectures, which have recently become popular in SR tasks. Specifically, it addresses a key limitation of transformer-based SR models: their restricted range of utilized information. To overcome this, the authors introduce a Hybrid Attention Transformer (HAT) block designed to activate and leverage more pixels by combining self-attention, channel attention, and a novel attention mechanism that uses overlapping windows during feature extraction.

The HAT architecture consists of three main parts. First, the input image passes through a convolutional layer to extract shallow features. These features are then processed by a series of Residual Hybrid Attention Groups (RHAGs) followed by another convolutional layer. Finally, a global residual connection fuses the shallow features from the first convolution with the deep features produced by the RHAGs, and a reconstruction module uses this combined information to generate the final high-resolution image.



![HAT architecture]({{ '/assets/images/01/HAT_architecture.png' | relative_url }})
{: style="max-width: 90%;"}
*Fig 1. HAT Architecture Overview [1].*

Each RHAG block is made up of hybrid attention blocks (HAB), an overlappong cross-attention block (OCAB) and a 3x3 convolution layer with a residual connection. 
#### Hybrid Attention Block (HAB)

One way the paper aims to increase the amount of activated pixels is by introducing a channel-attention-based (CAB) convolution block. 
This is inserted into a standard SWIN Transformer block after the first LayerNorm layer in parallel with the window-based  multi-head self-attention (W-MSA) module. For a given input feature X, HAB is computed as: 

$$
X_N = \mathrm{LN}(X),
$$

$$
X_M = (S)\mathrm{W\!-\!MSA}(X_N) + \alpha\,\mathrm{CAB}(X_N) + X,\tag{1}
$$

$$
Y = \mathrm{MLP}(\mathrm{LN}(X_M)) + X_M,
$$

$$X_N$$ and $$X_M$$ denote intermidiate features, Y is the output and $$\alpha$$ is added in order conflict of CAB and MSA. The self attention module W-MSA is caluculated by an input feature of size H X W X C, parititoned into $$\frac{HM}{M^2}$$ local windows then self-attention is applied to each window. Query, key and value matrices are computed by linear mappings Q,K and V for a local window feature. Window-based self attention is,

$$
Attention(Q,K,V) = SoftMax(QK^T /\sqrt{d} + B)V 
$$


with d being the dimension of query/key.

The CAB consists of the convolution layers wiht GELU activation, and a channel attention module. 


#### Overlapping Cross-Attention Block (OCAB)


![Overlapping window partition]({{ '/assets/images/01/overlapping.png' | relative_url }})
{: style="max-width: 100%;"}
*Fig 2. The overlapping window partition for OCA [1].*



An OCAB block is used to compute cross-window connections and enhance the representational ability of window-based self-attention. It consists of an overlapping cross-attention (OCA) layer followed by an MLP layer. Different window sizes are used to partition the projected features.

For input features $$X$$, let $$X_Q, X_K, X_V \in \mathbb{R}^{H \times W \times C}$$. $$X_Q$$ is partitioned into $$\frac{HW}{M^2}$$ non-overlapping windows of size $$M \times M$$. $$X_K$$ and $$X_V$$ are unfolded into $$\frac{HW}{M^2}$$ overlapping windows of size $$M_o\times M_o$$, where $$M_o$$ is calculated as

$$
M_o = (1+ γ) X M,
$$


where γ is a constant introduced in order to control overlapping size. 


#### All together

This results in a model that relies on a larger portion of the original image—making it less locally constrained—while remaining efficient by avoiding dense transformers and instead leveraging Swin attention and channel attention.



-adv: 
--designed to have a large receptive field
--uses shifted window transformers; takes advantage of transformers while limiting computational burden
-disadv: 
--implicit assumptions about noise, blurring, downsampling of image based on training data; may not match what is seen at inference time, leading to inaccuracy

### Look-Up Table (LUT) Methods

In the field of Super-Resolution (SR), there have been relatively few attempts to make SR practical for common consumer applications such as cameras, mobile phones, and televisions. Look-Up Table (LUT) methods aim to bridge this gap by introducing a single-image SR approach that runs significantly faster than traditional interpolation or deep neural network (DNN)–based methods. This efficiency is achieved by using a precomputed LUT, where the output values from an SR network are stored in advance. During inference, the system can quickly retrieve these high-resolution values by querying the LUT with low-resolution input pixels.


![LUT architecture]({{ '/assets/images/01/lut_architecture.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 1. LUT Method Overview [3].*



![LUT Comparison Table]({{ '/assets/images/01/lut_comparison.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 2. peak signal-to-noise ratio (PSNR) and runtime of various methods [3].*

To achieve this fast runtime, SR models are trained with a small receptive field, since the size of the SR-LUT grows exponentially with the receptive field size. This limitation introduces an inherent trade-off between PSNR and runtime: increasing the receptive field can improve reconstruction quality, but it also causes the LUT to expand dramatically, leading to slower performance.   

Specifically, the SR-LUT grows exponentially as given by:   


$$ \text{LUT Size} = (2^8)^{RF} \times r^2 \times 8\text{ bits} $$

Example with RF = 2 and r = 4:

$$
\begin{align}
\text{LUT Size}
&= (2^8)^2 \times 4^2 \times 8\ \text{bits} \\
&= 256^2 \times 16 \times 8\ \text{bits} \\
&= 65{,}536 \times 16 \times 1\ \text{byte} \\
&= 1{,}048{,}576\ \text{bytes} \\
&= 1\ \text{MB}
\end{align}
$$

The LUT stores precomputed output values for every possible combination of input pixels in a receptive field. Its size depends on the number of input pixels considered and the range of values each pixel can take. The LUT must cover all possible input cases and store the corresponding outputs.

After training the SR model, an SR-LUT table is created based on the dimensions of the receptive field (e.g., a 4D SR-LUT for an RF size of 4). The output values from the trained model are computed and stored in the LUT. During inference, input values are used as indices into the LUT, and the corresponding output values are retrieved. This allows super-resolution to be performed using only the LUT, without running the original model.

Currently, LUTs only work for fixed-scale images, which limits their real-world applicability when images are zoomed in or out. Recent extensions, such as IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution, propose frameworks for arbitrary-scale SR tasks. These methods adapt to diverse image structures, providing super-resolution across arbitrary scales while maintaining the efficiency that LUTs offer. 


TODO: CIte IM-LUT


-adv: 
--made to be fast and small
-disadv: 
--practically, very small receptive field; lookup table grows exponentially(?) with RF size
--implicit assumptions about noise, blurring, downsampling of image based on training data

### Unfolding Networks

The "unfolding" in "unfolding network" refers to splitting up (unfolding) the problem of de-degredation into two distinct subproblems, that being (1) unblurring and upsampling, and (2) denoising. With this approach, one can show that problem (1) has a closed-form optimal solution that can explicitly adapt to specific given types of degradation with 0 learned parameters; this greatly reduces the burden on the learned portion of the network, which now only needs to do denoising. The method is a kind of fusing of model-based and learning-based approaches; despite involving a variant of a UNet, it is designed to be zero-shot adaptable to any kind of degradation that is parameterized by a known blurring kernel, downsampling factor, and noise level.

(TODO: is the first-person "we", "our", etc wording appropriate for this, or should it be changed to third-person "they", "their", etc? the first person wording may give the wrong impression that we are trying to claim this method as our own or recreate it in some way, but it also feels weird to write proof-esque stuff in thrid person)

Now, we can go through and derive the method ourselves to illustrate how it arises:

The basic assumption will be that the input image for the method will be the blurred, downscaled, and additive white Gaussian noise-ed version of a ground-truth image, or in other words, our degraded image $$\vec y$$ is

$$
\vec y = (\vec x \otimes \vec k)\downarrow_s + \vec N
$$

where:
- $$\vec x \otimes \vec k$$ represents the application of blurring kernel $$\vec k$$ to ground-truth image $$\vec x$$ (via convolution)
- $$\downarrow_s$$ represents downsampling (decimation) by a factor of $$s$$
- $$\vec N \sim N(\vec 0, \sigma^2 I)$$ for $$\sigma \in \mathbb{R}^+$$

Given that $$\vec y \sim N((\vec x \otimes \vec k)\downarrow_s, \sigma^2 I)$$, its PDF is

$$
P(\vec y | \vec x) = \frac 1 {(2\pi\sigma^2)^{\frac d 2}}e^{-\frac 1 {2\sigma^2}||(\vec x \otimes \vec k)\downarrow_s - \vec y||_2^2}
$$

Furthermore, it will be useful to have an image prior, which we will define as

$$
\Phi(\vec x) = -\log P(\vec x)
$$

which will stand as a measure of how "natural" an image $$\vec x$$ is, ideally being minimised for any of our ground truth images and being maximised for images that are very noisy or unrealistic (i.e. unlike our GT images); we could interpret it simply as the negative logarithm of the a-priori probability distribution function of our ground truth images. In a practical sense, incorporating such a measure in our optimization will push our model toward creating images that conform to the patterns that typically appear in real images regarding color, brightness, etc, as opposed to overfitting on an image reconstruction loss function (in other words, a form of regularization).

Now, under a maximum a-posteriori (MAP) framework, our goal in super-resolution, given a degraded image $$\vec y$$ conforming to the assumptions above, can be defined as finding

$$
\hat x = \arg \max_{\vec x} P(\vec x | \vec y) \\
$$

Or, in other words, the clean image that the degraded image most likely started as. From Bayes theorem, we have 

$$
P(\vec x | \vec y) = \frac{P(\vec y | \vec x)P(\vec x)}{P(\vec y)} \propto P(\vec y | \vec x)P(\vec x)
$$

Since $$P(\vec y)$$ is constant with regard to $$\vec x$$. So now,

$$
\begin{aligned}
\hat x &= \arg \max_{\vec x} P(\vec x | \vec y) \\
&= \arg \max_{\vec x} P(\vec y | \vec x)P(\vec x) \\
&= \arg \min_{\vec x} -\log(P(\vec y | \vec x)P(\vec x)) = \arg \min_{\vec x} -\log P(\vec y | \vec x) - \log P(\vec x) \\
&= \arg \min_{\vec x} -(\log(\frac 1 {(2\pi\sigma^2)^{\frac d 2}}) - \frac 1 {2\sigma^2}||(\vec x \otimes \vec k)\downarrow_s - \vec y||_2^2) + \Phi(\vec x) \\
&= \arg \min_{\vec x} \frac 1 {2\sigma^2}||(\vec x \otimes \vec k)\downarrow_s - \vec y||_2^2 + \Phi(\vec x)
\end{aligned}
$$

Additionally, for practical reasons, we often include a "trade-off" hyperparameter, which we represent with $$\lambda$$, to balance the influence between our prior term $$\Phi$$ and our data term 
$$||(\vec x \otimes \vec k)\downarrow_s - \vec y||_2^2$$
, so we have

$$
\hat x = \arg \min_{\vec x} \frac 1 {2\sigma^2}||(\vec x \otimes \vec k)\downarrow_s - \vec y||_2^2 + \lambda\Phi(\vec x)
$$

Instead of directly minimizing this expression with something like gradient descent, one can actually notice that 
$$\arg \min_{\vec x} ||(\vec x \otimes \vec k)\downarrow_s - \vec y||_2^2$$
 has a closed form solution (which we will detail later); so, to take advantage of this fact, we can decouple the data term and prior term in our optimization by using half-quadratic splitting, where we introduce an auxiliary variable $$\vec z$$ into our optimization:

$$
\min_{\vec x, \vec z}\frac 1 {2\sigma^2}||(\vec z \otimes \vec k)\downarrow_s - \vec y||_2^2 + \lambda\Phi(\vec x) + \frac \mu 2 ||\vec x - \vec z||_2^2
$$

Now, $$\mu$$ is a hyperparameter controlling how much leeway we want to give $$\vec x$$ and $$\vec z$$ to be different, where $$\mu \to +\infty$$ recovers our original problem. Now, given that we have two optimization variables instead of just one, we can *unfold* the target and perform alternating iterative optimization (with $$j$$ as the step number):

$$
\begin{aligned}
\vec z_j &= \arg \min_{\vec z} \frac 1 {2\sigma^2}||(\vec z \otimes \vec k)\downarrow_s - \vec y||_2^2 + \frac \mu 2 ||\vec x_{j-1} - \vec z||_2^2 \\
\vec x_j &= \arg \min_{\vec x} \lambda\Phi(\vec x) + \frac \mu 2 ||\vec x - \vec z_j||_2^2
\end{aligned}
$$

Now, it may be useful to change the value of $$\mu$$ throughout our optimization; a small $$\mu$$ near the start of our optimization will help speed up convergence, and a large $$\mu$$ near the end will ensure that our solution actually corresponds to the original problem. So, define $$\mu_1, ..., \mu_J$$ for an optimization of $$J$$ steps, and let $$\alpha_j = \mu_j\sigma^2$$, so we have

$$
\begin{aligned}
\vec z_j &= \arg \min_{\vec z} ||(\vec z \otimes \vec k)\downarrow_s - \vec y||_2^2 + \alpha_j ||\vec x_{j-1} - \vec z||_2^2 \\
\vec x_j &= \arg \min_{\vec x} \Phi(\vec x) + \frac {\mu_j} {2\lambda} ||\vec x - \vec z_j||_2^2
\end{aligned}
$$

Now, as alluded, $$\vec z_j$$ actually still has a closed form solution:

[thing with fourier transforms]

The derivation of which is too long to include here, but is detailed in [4].

So, it remains to find 
$$\vec x_j = \arg \min_{\vec x} \Phi(\vec x) + \frac {\mu_j} {2\lambda} ||\vec x - \vec z_j||_2^2$$. 
One can notice that this is similar to our very first optimization target; indeed, finding $$\vec x_j$$ is equivalent to removing additive white Gaussian noise from $$\vec z_j$$ with $$\sigma^2_\text{noise} = \frac {\lambda} {\mu_j}$$ under a MAP framework. So, for convenience, we define 
$$\beta_j = \sqrt{\frac{\lambda}{\mu_j}}$$
 (the standard deviation of the Gaussian noise), and we have that finding $$x_j$$ is equivalent to a Gaussian denoising problem with noise level $$\beta_j$$ (or, $$\sigma^2_\text{noise} = \beta_j^2$$).

Given that this is a simple denoising task, we will opt for a denoising neural network. The paper in question uses a "ResUNet", which is a UNet with added residual blocks, similar to those from a ResNet. [more about the network structurt and a picture or something]

In order to ensure adaptibility and nonblind-ness of our method, it is useful to incorporate the noise level $$\beta_j$$ into the network input. The paper's method for doing this is fairly simple: given an input image $$3 \times H \times W$$, a constant matrix with size $$H \times W$$ with all entries equal to $$\beta_j$$ is appended on the channel dimension to create an input of shape $$4 \times H \times W$$, which is fed into the network as normal. 

[define the data and prior modules, talk about how prior module doesnt need to learn anything and can adapt to any kernel, sigma, and downsample factor (whcih is a good thing); talk about how removing implicit assumptions of kernel and noise levels and such makes the model more generalizable]

[talk about training process]

[talk about results]

[include more images somewhere]

-adv: 
--nonblind; explicitly adapts for diff blurring kernels, amts of noise, and downsampling factors
--limits learned parameters by only using learning for denoising step; deblurring and upsampling is closed-form, but still included in the chain of the model to allow learning to take advantage of it
-disadv: 
--sequential design w/ conv nets and (inverse) FFTs may be slow? 
--use of only conv nets for denoising step may limit effective receptive field

-more classical method can better adapt to diff blurring kernel, etc without extensive training; balance of learning based and model based method is good idk

## An Extension of the Hybrid Attention Transformer

### Adaptive Sparse Transformer (AST)

this modifies the attention matrix AND the FFN after the values are computed from the self attention layer; potentially modifying a lot of parameters

### Putting it Together

idea: add AST to OCA and increase size of overlapping K/V windows, since the model can do a better job at filtering out any noise that it introduces; this part of the model is also like an overall filter for each RHAG block, which AST may be good for?

### Results


## References

[1] Chen, Xiangyu, et al. “Activating More Pixels in Image Super-Resolution Transformer.” arXiv preprint arXiv:2205.04437, 2023.

[2] Zhang, Kai, et al. “Deep Unfolding Network for Image Super-Resolution.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 3217–3226.

[3] Jo, Younghyun, and Seon Joo Kim. “Practical Single-Image Super-Resolution Using Look-Up Table.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 691–700. doi:10.1109/CVPR46437.2021.00075.

[4] N. Zhao, Q. Wei, A. Basarab, N. Dobigeon, D. Kouamé and J. -Y. Tourneret, "Fast Single Image Super-Resolution Using a New Analytical Solution for ℓ2 – ℓ2 Problems," in IEEE Transactions on Image Processing, vol. 25, no. 8, pp. 3683-3697, Aug. 2016



---
