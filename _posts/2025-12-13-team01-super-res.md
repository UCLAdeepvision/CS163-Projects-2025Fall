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

## Background

-super res encompasses upscaling, denoising, deblurring
-mention classical methods

## A Survey of Deep Learning for Super Resolution

### Hyrbid Attention Transformer (HAT)

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

-adv: 
--made to be fast and small
-disadv: 
--practically, very small receptive field; lookup table grows exponentially(?) with RF size
--implicit assumptions about noise, blurring, downsampling of image based on training data

### Unfolding Methods

[introduction, introduce the paper and how it combines learning-based and model-based methods]

The "unfolding" part of "unfolding method" comes from the way that we deal with our optimization target: our basic assumption will be that the input image for our method will be the blurred, downscaled, and additive white Gaussian noise-ed version of a ground-truth image, or in other words:

$$
\vec y = (\vec x \otimes \vec k)\downarrow_s + \vec N
$$

where:
- $$\vec x \otimes \vec k$$ represents the application of blurring kernel $$vec k$$ to ground-truth image $$\vec x$$
- $$\downarrow_s$$ represents downsampling (decimation) by a factor of $$s$$
- $$\vec N \sim N(\vec 0, \sigma^2 I)$$ for $$\sigma \in \mathbb{R}$$

Given that $$\vec y \sim N((\vec x \otimes \vec k)\downarrow_s, \sigma^2 I)$$, its PDF is

$$
P(\vec y | \vec x) = \frac 1 {(2\pi\sigma^2)^{\frac d 2}}e^{-\frac 1 {2\sigma^2}||(\vec x \otimes \vec k)\downarrow_s - \vec y||_2^2}
$$

Furthermore, it will be useful to have an image prior, which, for a hyperparameter $$\lambda$$, we will define as

$$
\Phi(\vec x) = -\frac{1}{\lambda}\log P(\vec x)
$$

which will stand as a measure of how "natural" an image $$\vec x$$ is, ideally being minimised for any of our ground truth images and being maximised for images that are very noisy or unrealistic (i.e. unlike our GT images); we could interpret it simply as the scaled negative logarithm of the a-priori probability distribution of our ground truth images. In a practical sense, incorporating such a measure in our optimization will push our model toward creating images that conform to the patterns that typically appear in real images regarding color, brightness, etc, as opposed to overfitting on an image reconstruction loss function (in other words, a form of regularization).

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
&= \arg \min_{\vec x} -(\log(\frac 1 {(2\pi\sigma^2)^{\frac d 2}}) - \frac 1 {2\sigma^2}||(\vec x \otimes \vec k)\downarrow_s - \vec y||_2^2) + \lambda\Phi(\vec x) \\
&= \arg \min_{\vec x} \frac 1 {2\sigma^2}||(\vec x \otimes \vec k)\downarrow_s - \vec y||_2^2 + \lambda\Phi(\vec x)
\end{aligned}
$$

Instead of directly minimizing this expression with something like gradient descent, one can actually notice that 
$$\arg \min_{\vec x} ||(\vec x \otimes \vec k)\downarrow_s - \vec y||_2^2$$
 has a closed form solution (which we will detail later), so we can decouple the data term and prior term in our optimization by using half-quadratic splitting, where we introduce an auxiliary variable $$\vec z$$ into our optimization:

$$
\min_{\vec x, \vec z}\frac 1 {2\sigma^2}||(\vec z \otimes \vec k)\downarrow_s - \vec y||_2^2 + \lambda\Phi(\vec x) + \frac \mu 2 ||\vec x - \vec z||_2^2
$$

Now, $$\mu$$ is a hyperparameter controlling how much leeway we want to give $$\vec x$$ and $$\vec z$$ to be different, where $$\mu \to +\infty$$ recovers our original target. Now, given that we have two optimization variables instead of just one, we can *unfold* the target and perform iterative optimization (with $$j$$ as the step number):

$$
\begin{aligned}
\vec z_j &= \arg \min_{\vec z} \frac 1 {2\sigma^2}||(\vec z \otimes \vec k)\downarrow_s - \vec y||_2^2 + \frac \mu 2 ||\vec x_{j-1} - \vec z||_2^2\\
&= \arg \min_{\vec z} ||(\vec z \otimes \vec k)\downarrow_s - \vec y||_2^2 + \mu\sigma^2 ||\vec x_{j-1} - \vec z||_2^2 \\
\vec x_j &= \arg \min_{\vec x} \lambda\Phi(\vec x) + \frac \mu 2 ||\vec x - \vec z_j||_2^2
& \\
\end{aligned}
$$

Now, it may be useful to change the value of $$\mu$$ throughout our optimization as a form of gradual regularization; a small $$\mu$$ near the start of our optimization will help speed up the process, and a large $$\mu$$ near the end will ensure that our solution actually corresponds to the original problem. So, define $$\mu_1, ..., \mu_J$$ for an optimization of $$J$$ steps, and let $$\alpha_j = \mu_j\sigma^2$$, so we have

$$
\begin{aligned}
\vec z_j &= \arg \min_{\vec z} ||(\vec z \otimes \vec k)\downarrow_s - \vec y||_2^2 + \alpha_j ||\vec x_{j-1} - \vec z||_2^2 \\
\vec x_j &= \arg \min_{\vec x} \lambda\Phi(\vec x) + \frac {\mu_j} 2 ||\vec x - \vec z_j||_2^2
& \\
\end{aligned}
$$

Now, as alluded, $$\vec z_j$$ still has a closed form solution:

[thing with fourier transforms]

The derivation of which is too long to include here, but is detailed in [number for reference to https://ieeexplore.ieee.org/document/7468504]

So, it remains to find 
$$\vec x_j = \arg \min_{\vec x} \lambda\Phi(\vec x) + \frac {\mu_j} 2 ||\vec x - \vec z_j||_2^2$$. 
One can notice that this is similar to our very first optimization target; indeed, finding $$\vec x_j$$ is equivalent to removing additive white Gaussian noise from $$\vec z_j$$ with $$\sigma^2 = \frac 1 {\mu_j}$$ under a MAP framework. However, if we assume that $$-\log P(\vec x) \approx \Phi(\vec x)$$, then we can instead use $$\lambda$$ as a hyperparameter further balancing the influence between the prior term $$\Phi(\vec x)$$ and the data term $$||\vec x - \vec z_j||_2^2$$:

$$
\vec x_j = \arg \min_{\vec x} \Phi(\vec x) + \frac {\mu_j} {2\lambda} ||\vec x - \vec z_j||_2^2
$$

Now, this corresponds to removing additive white Gaussian noise from $$z_j$$ with $$\sigma^2 = \frac{\lambda}{\mu_j}$$ under a MAP framework. For convenience, we define 
$$\beta_j = \sqrt{\frac{\lambda}{\mu_j}}$$
 (the standard deviation of the Gaussian noise), so our optimization is now

$$
\begin{aligned}
\vec z_j &= \arg \min_{\vec z} ||(\vec z \otimes \vec k)\downarrow_s - \vec y||_2^2 + \alpha_j ||\vec x_{j-1} - \vec z||_2^2 \\
\vec x_j &= \arg \min_{\vec x} \Phi(\vec x) + \frac {1} {2\beta_j^2} ||\vec x - \vec z_j||_2^2
& \\
\end{aligned}
$$

With all of that, it still remains to find a method for updating $$\vec x$$ at each step; given that this is a simple denoising task, we will opt for a denoising neural network. The paper in question uses a "ResUNet", which is a UNet with added residual blocks, similar to those from a ResNet. [more about the network structurt and a picture or something]

In order to ensure adaptibility of our method, we will incorporate the noise level $$\beta_j$$ into our network input. To this end, given an input image $$3 \times H \times W$$, a constant matrix with size $$H \times W$$ with all entries equal to $$\beta_j$$ is appended to the channel dimension to create an input of shape $$4 \times H \times W$$, which is fed into the network as normal. 

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



---
