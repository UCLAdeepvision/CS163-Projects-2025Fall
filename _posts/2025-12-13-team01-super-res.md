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

Super-resolution is a natural problem for image-to-image methods in computer vision, referring to the process of using a degraded, downsampled image to recover the original image before degradation and downsampling. Of course, this is an ill-posed problem; two different high-resolution images can downsample and degrade into the same low-resolution image (that is, the process is not injective), so truly inverting the process is impossible. For this reason, we settle for the task of estimating a function that approximates an inverse, producing a high-resolution image based solely on some mathematical assumptions and the information of the low-resolution image. This includes simple, classical methods, such as nearest-neighbor or bicubic upsampling, as well as, in recent years, statistical methods making use of deep learning for computer vision.

We will be surveying three of these recent methods, highlighting the very different ways that one can go about achieving the same end goal in super-resolution:

- The Hybrid Attention Transformer (HAT), a shifted-window transformer-based method

- Look-Up Table (LUT) methods, which use precomputed table of pixel estimates

- Unfolding networks, which combine classical model-based methods with learning-based methods

Additionally, we will be discussing an experiment that we carried out involving a modification to the HAT architecture.

## A Survey of Deep Learning for Super Resolution

### Hybrid Attention Transformer (HAT)

The Hybrid Attention Transformer aims to advance the field of low-level super-resolution by improving transformer architectures, which have recently become popular in SR tasks. Specifically, it addresses a key limitation of transformer-based SR models: their restricted receptive field. To overcome this, [1] introduces a Hybrid Attention Transformer (HAT) architecture designed to leverage more image pixels for reconstruction by combining self-attention, channel attention, and a novel overlapping-window attention mechanism.

<!--i feel like this high-level overview does not make sense to a reader who doesnt already know what a RHAG or HAB or OCAB is-->
<!-- The high-level overview of HAT is as follows: first, the input image passes through a convolutional layer to extract shallow features. These features are then processed by a series of Residual Hybrid Attention Groups (RHAGs) followed by another convolutional layer; each RHAG block is made up of hybrid attention blocks (HAB), an overlapping cross-attention block (OCAB) and a convolution layer with a residual connection (these modules are described in detail below). Finally, a global residual connection combines the shallow features from the first convolution with the deep features produced by the RHAGs, and a reconstruction module featuring a pixel-shuffle uses this combined information to generate the final high-resolution image. -->

![HAT architecture]({{ '/assets/images/01/HAT_architecture.png' | relative_url }})
{: style="max-width: 90%;"}
*Fig 1. HAT Architecture Overview [1].*

#### Hybrid Attention Block (HAB)

![hybrid attention block]({{ '/assets/images/01/hab.png' | relative_url }})
{: style="max-width: 90%;"}
*Fig 2. Hybrid Attention Block [1].*

The Hybrid Attention Block (HAB) aims to combine the deep flexibility of vision transformers with the efficiency and global accessibility of channel attention.

![channel attention block]({{ '/assets/images/01/cab.png' | relative_url }})
{: style="max-width: 90%;"}
*Fig 3. Channel Attention Block [1].*

The Channel Attention Block (CAB) consists of two convolutional layers separated by a GELU activation, followed by a channel attention (CA) module. The CA module consists of global average pooling (GAP), followed by two $$1 \times 1$$ convolutions, ensuring that the channel number is the same as prior to the CA module, separated by another GELU. Then, the sigmoid function is applied to the now-$$1 \times 1 \times C$$-shaped output to create channel weights, which are multiplied channel-wise by the pre-CA module input, to get an output of the same $$H \times W \times C$$ shape, where each channel has been scaled by a factor in $$[0, 1]$$. The factors are applied globally to each channel and are determined by GAP + convolutions, allowing all channels to utilize information from, to some degree, all positions in all other channels in an efficient manner. 

In the HAB, the CAB is inserted into a standard Swin transformer block after the first LayerNorm in parallel with the window-based multi-head self-attention (W-MSA) module, where they are then combined additively (along with a residual connection). The efficiency and simplicity of the CAB comes at the cost of specific positional information being unutilized due to GAP, which is why the CAB is used in parallel with W-MSA, which explicitly *does* account for positional information. So, then, for a given input feature $$X$$, HAB is computed as: 

$$
X_N = \mathrm{LN}(X),
$$

$$
X_M = (\text{S})\text{W-MSA}(X_N) + \alpha\,\mathrm{CAB}(X_N) + X
$$

$$
Y = \mathrm{MLP}(\mathrm{LN}(X_M)) + X_M,
$$

where LN is the LayerNorm, (S)W-MSA is the (shifted) window-multihead self attention, MLP is the standard positionwise feed-forward network of a transformer, $$X_N$$ and $$X_M$$ denote intermediate features, $$Y$$ is the output, and $$\alpha$$ is a small hyperparameter that tempers the influence of the CAB compared to the W-MSA, to avoid issues of conflict between the two modules during optimization; empirically, $$\alpha=0.01$$ was found to give the best performance, as this parallel scheme can introduce significant stability issues in the optimization process if the influence of each branch is not controlled. Within-window linear mappings are used to get $$Q$$, $$K$$, and $$V$$ matrices, though, notably, these mappings are applied on the individual spatial pixels of the input, not on patches (that is, patches of size $$1 \times 1$$ are used), as the spatial dimensions of the input need to be maintained throughout the network to account for residual connections. A fixed window size is employed, the same as in the regular Swin transformer, however, notably, the model does not contain any patch merging, again in service of retaining the shape of the image for residual connections. Within-window attention uses the standard attention scheme, with a fixed sinusoidal relative positional encoding within each window similar to that of "Attention is All You Need".

#### Overlapping Cross-Attention Block (OCAB)

![Overlapping window partition]({{ '/assets/images/01/overlapping.png' | relative_url }})
{: style="max-width: 100%;"}
*Fig 4. The overlapping window partition for OCA [1].*

OCAB modules are used throughout the HAT to further allow for cross-window connections; the basic idea is that windowed self-attention is performed on the input as normal, in the exact same manner as in the HAB module, though the windows for the keys and values, while still centered on the same locations as the windows of the queries, are larger than those of the queries, allowing each window to pull in some information from neighboring windows that it normally would not see.

Specifically, for input features $$X$$, let $$X_Q, X_K, X_V \in \mathbb{R}^{H \times W \times C}$$. $$X_Q$$ is partitioned into $$\frac{HW}{M^2}$$ non-overlapping windows of size $$M \times M$$. $$X_K$$ and $$X_V$$ are partitioned into $$\frac{HW}{M^2}$$ overlapping windows of size $$M_o\times M_o$$, where $$M_o$$ is calculated as

$$
M_o = (1 + \gamma)M,
$$

where $$\gamma$$ is a hyperparameter introduced in order to control the size of the overlapping windows, and the input is zero-padded to account for this larger window. Aside from shifting windows, this allows for even more cross-window information transmission, as each query window, which are the standard size, can pull in information from the key and value windows that include the query window and extend into the space that would otherwise be a separate, independent window. Additionally, OCAB is specifically designed to mitigate the blocking artifacts that the standard Swin transformer produces:

![Blocking ]({{ '/assets/images/01/blocking.png' | relative_url }})
{: style="max-width: 100%;"}
*Fig 5. Comparison of blocking between SwinIR and HAT [1].*

The OCAB combats this by directly stepping over the hard boundaries between windows, reducing their presence in the extracted features; each query window has access to information about the space beyond its own boundaries, so they can better create smooth boundaries in the features between adjacent windows. While standard window shifting has a similar goal, each shifted window still only has access to its own pixels, which may still lead to hard boundaries and blocking between windows that become present in the features.

Empirically, $$\gamma=0.5$$ was found to give the best performance, translating to overlapping windows that are, by area, 4/9 composed of the original window and 5/9 composed of adjacent windows. This is less extreme than the shifted windows of the standard Swin transformer, where, since the regular shifting amount is half of the window size in each dimension, shifted windows are only 1/4 composed of their original window, and are 3/4 composed of other windows. Still, this additional method of cross-window attention gives the model greater flexibility and another opportunity to use a larger portion of the original image in its reconstruction process.

#### Residual Hybrid Attention Group (RHAG)

![Residual Hybrid Attention Group]({{ '/assets/images/01/rhag.png' | relative_url }})
{: style="max-width: 100%;"}
*Fig 6. The structure of a Residual Hybrid Attention Group [1].*

These modules are combined in a further modular format via a Residual Hybrid Attention Group (RHAG), which puts a variable number of HAB modules in sequence which are followed by an OCAB module, with a final convolutional layer to ensure compatability between shapes of the input and output that allows for a residual connection over the entire module. The sequencing of HAB modules mirrors the structure of standard Swin stages, as the attention windows are shifted between subsequent HABs; though, patch merging is skipped in favor of the OCABs, as they both aim to combine information between windows, and OCAB can retain the spatial size of the input.

#### High-Level Model Structure

![High Level HAT]({{ '/assets/images/01/highlevelhat.png' | relative_url }})
{: style="max-width: 100%;"}
*Fig 7. The high-level structure of the HAT [1].*

Contrary to the standard Swin, projection to the desired channel dimension is done using an initial convolutional layer; though, mirroring the structure of the Swin transformer, multiple RHAG modules are placed in sequence (similar to the sequence of stages in Swin), though in this case, a residual connection (along with a convolutional layer to ensure compatability of shapes) is employed across the series of RHAGs. Then, the final image reconstruction is performed using convolutions (for channel number adjustments) and pixel shuffle, where pixels from many channels are pulled together sequentially into a smaller number of channels with a larger image size.

#### Results

![LAM Results]({{ '/assets/images/01/lam.png' | relative_url }})
{: style="max-width: 100%;"}
*Fig 8. LAM results for different SR methods [1].*

In practice, the HAT does, indeed, use a larger portion of the input image for reconstruction than other transformer-based SR methods. Above is a comparison of Local Attribution Maps (LAM) between different methods, which is a method for measuring which portions of the input were most influential for a given patch of the output. We see that HAT is highly nonlocal compared to other methods in terms of LAM, and performs better for it.

![Quantitative Results]({{ '/assets/images/01/quantitativeresultshat.png' | relative_url }})
{: style="max-width: 100%;"}
*Fig 9. Quantitative results for HAT compared to other methods [1].*

Quantitatively, in terms of peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM), two standard metrics for image reconstruction, HAT outperforms its peers (if only slightly).

The HAT is a notable modification of the Swin transformer architecture, designed for super resolution, due to its large practical receptive field; while all SR methods have an entire image to work with as input, it takes effort and specific design to actually use the information that is there in an efficient manner, which can be crucial to reconstructing an image correctly. Judging by its quantitative results, the future state-of-the-art in SR will likely be driven, to some degree, by the development of methods that are designed to produce even larger receptive fields.

<!-- -adv: 
--designed to have a large receptive field
--uses shifted window transformers; takes advantage of transformers while limiting computational burden
-disadv: 
--implicit assumptions about noise, blurring, downsampling of image based on training data; may not match what is seen at inference time, leading to inaccuracy -->

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


<!-- -adv: 
--made to be fast and small
-disadv: 
--practically, very small receptive field; lookup table grows exponentially(?) with RF size
--implicit assumptions about noise, blurring, downsampling of image based on training data -->

### Unfolding Networks

The "unfolding" in "unfolding network" refers to splitting up the problem of de-degredation into two distinct subproblems, that being (1) unblurring and upsampling, and (2) denoising. With this approach, one can show that problem (1) has a closed-form optimal solution that can explicitly adapt to specific given types of degradation with 0 learned parameters; this greatly reduces the burden on the learned portion of the network, which now only needs to do denoising. The method is a kind of fusing of model-based and learning-based approaches; despite involving a variant of a UNet, it is designed to be zero-shot adaptable to any kind of degradation that is parameterized by a known blurring kernel, downsampling factor, and noise level.

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

<!-- -adv: 
--nonblind; explicitly adapts for diff blurring kernels, amts of noise, and downsampling factors
--limits learned parameters by only using learning for denoising step; deblurring and upsampling is closed-form, but still included in the chain of the model to allow learning to take advantage of it
-disadv: 
--sequential design w/ conv nets and (inverse) FFTs may be slow? 
--use of only conv nets for denoising step may limit effective receptive field

-more classical method can better adapt to diff blurring kernel, etc without extensive training; balance of learning based and model based method is good idk -->

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
