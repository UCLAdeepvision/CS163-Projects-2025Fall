---
layout: post
comments: true
title: super res title
author: Thomas Peeler, Dylan Truong, Asher Christian, Daniel Chvat
date: 2025-01-01
---


> Abstract: survey of 3 super resolution models, modification of one of them; 3 models highlight different ways of doing super resolution

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

The authors compared 3 versions of the model: HAT-S, HAT, and HAT-L: HAT has a similar size to the existing Swin super-resolution model SwinIR, with 6 RHAG modules with 6 HABs each; HAT-L is larger than HAT, with 12 RHAGs instead of 6; HAT-S is smaller than HAT, with depthwise convolutions and a smaller channel number throughout the model, resulting in overall computation similar to that of SwinIR. Quantitatively, in terms of peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM), two standard metrics for image reconstruction, HAT's variants outperform their peers (if only slightly), and larger HAT models tend to yield better performance (again, if only slightly).

The HAT is a notable modification of the Swin transformer architecture, designed for super resolution, due to its large practical receptive field; while all SR methods have an entire image to work with as input, it takes effort and specific design to actually use the information that is there in an efficient manner, which can be crucial to reconstructing an image correctly. Judging by its quantitative results, the future state-of-the-art in SR will likely be driven, to some degree, by the development of methods that are designed to produce even larger receptive fields.

<!-- -adv: 
--designed to have a large receptive field
--uses shifted window transformers; takes advantage of transformers while limiting computational burden
-disadv: 
--implicit assumptions about noise, blurring, downsampling of image based on training data; may not match what is seen at inference time, leading to inaccuracy -->

### Look-Up Table (LUT) Methods

There have been relatively few attempts to make modern learning-based methods of super-resolution practical for common consumer applications such as cameras, mobile phones, and televisions. Look-Up Table (LUT) methods, as described in [3], aim to bridge this gap by using a precomputed LUT, where the output values from an SR network are stored in advance; during inference, the system can quickly retrieve these high-resolution values by querying the LUT with low-resolution input pixels.

![LUT architecture]({{ '/assets/images/01/lut_architecture.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 1. LUT Method Overview [3].*

#### The LUT

Although it may seem backwards, it may be useful to start with how the LUT will be used in inference: given a low-resolution image and an upscaling factor $$r$$, our goal is will be to map each pixel of the low-res image to an image patch of size $$r \times r$$, which we then assemble into a full upscaled image. This process will be done independently for each color channel, so that we are assembling 3 independent images with 1 color channel each and then concatenating them to form a 3-channel image. For a given image pixel at location $$(i, j)$$, we would like this mapping to take into account the pixels surrounding $$(i, j)$$ as well as the pixel itself; so, to get the patch for pixel $$(i, j)$$, the LUT will be indexed by a fixed number of pixel values surrounding $$(i, j)$$, and at this index, there will be $$r^2$$ values that we can use to construct an image patch (given that we use pixel values for indexing, we use 8-bit color). For instance, for $$r=2$$ and a LUT receptive field of 4, let 
$$I_0 = image[i, j], I_1 = image[i+1, j], I_2 = image[i, j+1], I_3 = image[i+1, j+1]$$. 
Then, $$LUT[I_0][I_1][I_2][I_3][0][0]$$ will contain the top-left corner of the upscaled patch, and $$LUT[I_0][I_1][I_2][I_3][1][1]$$ will contain the bottom-right corner. This will be the basic method that the LUT uses in inference, which only requires table lookup, and no network computation.

#### The SR Network

Now, with a clear goal of how the LUT should function, we can construct the network that will give us its values. Given that we want the patch in the LUT for each specific pixel value to depend solely on a fixed number of surrounding pixels, a convolutional network with a finely-controlled receptive field seems natural; again, note that the LUT upscaling process is performed independently on each color channel, so the input to this network will only have 1 channel. Specifically, given a receptive field size $$n$$ and an upscaling factor $$r$$, we can construct a network as:

- a series of convolutional layers (with nonlinearities), where, at the end, each output pixel has a gross receptive field of size $$n$$ and the output has $$r^2$$ channels

- a pixel shuffle that will construct upscaled patches from our channels and concatenate them together into a full upscaled image

with this, we have constructed a network where each tiled $$r \times r$$ patch of the output depends only on a (contiguous) size $$n$$ receptive field from the input. Now, if we optimize this network for super-resolution, to construct our LUT, we can simply generate patches of size $$n$$ (or more to create an array) with specific values (which will be indexes in the LUT) and feed them into the network, and the resulting upscaled patch can be added to the LUT. The kernels of the convolutional layers could be anything as long as it conforms to the receptive field requirement, however, the authors of the paper use an architecture where the first convolutional layer results in a receptive field of size $$n$$, and all subsequent layers are $$1\times 1$$ convolutions (the issue of the kernel reducing the size of the input is dealt with by pre-padding the image); additionally, they use a square (or anything closest to square) kernel when possible, as it will capture the pixels that are most adjacent, and thus most relevant, to each other.

#### LUT Size

Of course, the principal design priority of the LUT is efficiency in speed and size. Using 8-bit color, if we construct a LUT that accounts for every possible input patch, one can compute the size of the LUT for a given receptive field $$n$$ and upscale factor $$r$$ as 

$$
(2^8)^n r^2 \text{ bytes}
$$

This can be seen clearly from the structure of the LUT; there are $$2^8$$ possible values for a given 1-channel pixel, so for a RF of size $$n$$, there are $$(2^8)^n$$ possible input patches. At each index corresponding to an input patch, there is an upscaled patch of size $$r^2$$, whose entries are also each 8 bits (or 1 byte). Clearly, this will not scale well with $$n$$, and in fact, $$n=4$$ and $$r=2$$ gives us a LUT that is 16GB large. To counteract this, the paper uses a "sampled LUT", where only a subset of the possible input patches are sampled for use in the LUT; specifically, they found that sampling the color space of $$0$$ to $$2^8-1$$ uniformly (including endpoints) with a sampling interval of $$2^4$$ over each pixel of the input patch worked well to reduce the size of the LUT while retaining good performance, reducing the 16GB LUT to just $$(1+\frac{2^8}{2^4})^4 2^2 = 334084$$ bytes, or about 326KB.

![LUT sizes]({{ '/assets/images/01/lutsizes.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 2. A comparison of LUT sizes [3].*

Of course, this sampled LUT introduces the issue of indexing: how do we index from the LUT if the input patch doesn't match a patch that was seen while filling out the table? For this, the authors used interpolation; instead of indexing one upscaled patch, we can interpolate between up to $$2^n$$ patches, for a RF size $$n$$. For instance, for a LUT where $$n=2$$, using a sampling interval of $$2^4$$, getting the upscaled patch for an input patch where $$I_0 = 8, I_1 = 8$$ would require interpolating between $$LUT[0][0], LUT[0][1], LUT[1][0], LUT[1][1]$$. Using this sampling interval, these nearest points can actually be found easily by looking at the bit representation of $$I_j$$: the first 4 bits, when converted to integer, will be the index of the lower part of the interpolation (e.g. for $$I_j = 20$$, 20 $$\to$$ 00010100, and then 0001 $$\to$$ 1, meaning that $$LUT[1]$$ will be the lower part of the interpolation and $$LUT[2]$$ will be the upper part); this method of indexing for the interpolation points adds even further to the LUT's speed. The issue of interpolating between points in $$n$$ dimensional space is complex in its own right, though we note that the authors found that linear interpolation was slower than triangular interpolation (and its higher-dimensional couterparts), especially for larger $$n$$.

![LUT sizes]({{ '/assets/images/01/interpolation.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 3. A comparison of interpolation methods for $$n=2, 3, 4$$ [3].*

#### Rotational Ensemble

Additionally, during training and testing, the authors employed the strategy of "rotational ensemble". This is a method of data augmentation that involves:

- rotating each input by 90, 180, and 270 (and 0) degrees

- feeding each of 4 rotations of the input into the model

- rotating each the super-resolution outputs such that they are all at their original orientation

- taking the average over the 4 outputs, and treating that as your final output

This method is well-suited to super resolution, as the semantic meaning of the image is not important; images that are rotated still fall under our base assumption of what "real" images look like for super resolution, so the model can improve by optimizing over them. In our case, this method can actually improve the effective receptive field of the model substantially:

![Rotational Ensemble Illustration]({{ '/assets/images/01/rotensemble.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 4. An illustration of the effect of rotational ensemble on the receptive field; a 2x2 RF is effectively increased to a 3x3 RF [3].*

During training, rotational ensemble simply works well as a way to increase your dataset size. However, this method can be used in inference as well; we can construct 4 different super-resolution predictions using the LUT, and then average them together. For a 2x2 square RF, one can see, in the illustration above representing 2x upsampling, that the relative position of the patch in the output corresponding to the RF in the input will be determined by the relative position of the top-left pixel of the RF in the input image; that is, any RF in the input, no matter the rotation of, will correspond to the same patch in the output as long as the top-left pixel of the RF is the same across those RFs in the input. So, when we look at the RFs across our rotational ensemble when the top-left pixel is kept constant, we see that their union will be a 3x3 area in the input, which means that the given output patch will be influenced by all pixels in the 3x3 area, effectively increasing our inference RF size to 9.

While this does increase our computation time by a factor of 4 (discounting rotations and averaging), it is likely a worthy trade-off for such a substantial increase in RF size, which would normally require increasing the LUT size to an unmanageable degree.

#### Results

![LUT Comparison Table]({{ '/assets/images/01/lut_comparison.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 4. peak signal-to-noise ratio (PSNR) and runtime of various methods for 320 x 180 -> 1280 x 720 upsampling [3].*

There are 3 models mentioned in the paper: V, F, and S, which have a receptive field of 2, 3, and 4, respectively. F and S use a sampled LUT with a sampling interval of $$2^4$$, while V uses a full LUT. We see that the LUT methods provide a good tradeoff in terms of speed vs reconstruction quality, where F is even faster than bicubic interpolation, and V is even faster than bilinear interpolation.

![LUT qualitative results]({{ '/assets/images/01/lutqualitative.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 4. Qualitative comparison of results [3].*

![LUT qualitative results]({{ '/assets/images/01/lutquantitative.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 5. Quantitative comparison of results; runtimes for all methods besides sparse coding are on a Galaxy S7 phone [3].*

We see that, quantitatively, all LUT variants perform substantially better than traditional interpolation upsampling methods, and are competitive with methods that require far greater runtime and/or storage space. Interestingly, there are some cases where the F model seems to perform quantitatively better than the S model, despite the fact that the only difference between them is that S has a larger receptive field. This likely just indicates a plateau in performance wrt receptive field size, or the differently shaped receptive fields may fare differently on different datasets (since RF=3 for F and RF=4 for S, the RF for F is a line and the RF for S is a square).

LUT methods are an interesting way to make super-resolution more efficient; instead of making a more efficient network, it avoids using a network for inference altogether. However, there are some obvious flaws: the quantitative results do not make it state-of-the-art in that regard, and the methodology of using the same LUT for each color channel independently seems somewhat unintuitive. Additionally, a given LUT is confined to a fixed receptive field size and upscaling factor; an entirely new SR network and table must be created to change these factors, which restricts the method's practical use.

#### IM-LUT

Since its invention in 2021, there have been numerous published papers for extensions and modifications to the LUT. One of these, published in 2025, is Interpolation Mixing LUT, or IM-LUT [5]. IM-LUT allows for the model to adapt to the scaling factor at inference time, meaning that a single IM-LUT can be used to scale any image to any size. This is achieved by first upscaling the input image with standard interpolation algorithms, such as bicubic upsampling, and then using a standard LUT on the upscaled image. However, the IM-LUT can also adapt its interpolation upscaling to the input image; a single IM-LUT uses many different interpolation algorithms to upscale the input image, and then uses a weighted average, with weights based on the image itself and the given scale factor, to combine the interpolations together before the final refinement LUT. With this, the IM-LUT can adapt to any scaling factor while still also dynamically adapting to the specific input image.

Now, we go through the structure of the IM-LUT

##### IM-Net

![IM Net]({{ '/assets/images/01/imnet.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 5. IM-Net, the network used to generate the LUTs for IM-LUT [3].*

There are two overall parts to the IM-Net: weight map prediction, and image refinement. Image refinement is simply a network that refines an the interpolated input image; it takes an image of any given size, and produces an image of the same size. This part of the network is simple, but the weight maps are more involved:

The IM-LUT works with a fixed set of $$K$$ interpolation functions, and for any given image, the network predicts per-pixel weights for each interpolation function; these weight maps allow the network to take advantage of the fact that different interpolation methods may fare better in different areas of a given image. The weight maps are a combination of the output of two other networks:

- An image-to-weight map network, that takes the low-resolution image and produces low-resolution weight maps.

- An upscaling factor-to-scale vector network ("scale modulator"), that takes the scaling factor as input and outputs a vector of per-interpolation function weights.

The weights from the scale modulator are multiplied by the initial weight maps (one scale weight per weight map) to produce the final weight maps. The networks optimize their predictions of these weight maps as so:

- For a given image (where the ground truth is known) and upscaling factor, downscale the image and then get interpolations back to the original size using each given interpolation function.

- Calculate per-interpolation deviations from the GT image using per-pixel L1 loss, i.e. 
$$E_k[i, j] = |I_{\text{interp } k}[i, j] - I_{\text{GT}}[i, j]|$$ 
for $$k=1,...,K$$.

- Compute ground-truth weight maps as a temperature-controlled per-pixel softmax over all negative interpolation deviations, i.e.
$$W^{GT}_k[i, j] = \frac{e^{-\beta E_k[i, j]}}{\sum_{\ell=1}^K e^{-\beta E_\ell[i, j]}}$$ 
for $$\beta$$ being the temperature constant.

- Compute the "guide" loss (or, the weight map loss) as the L2 reconstruction loss between the predicted and ground-truth weight maps, that is
$$\mathcal{L}_{\text{guide}} = \sum_{\ell=1}^K ||\hat W_\ell - W^{GT}_\ell||_2$$

Then, when the predicted weight maps are used to create a final interpolated image, and that is run through the refinement network, the final reconstruction loss can be found as 
$$\mathcal{L}_{\text{rec}} = ||\hat I - I^{GT}||_2$$ 
and the final loss that we optimize over is 
$$\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda \mathcal{L}_{\text{guide}}$$ 
where $$\lambda$$ is our trade-off hyperparameter. With this, we can optimize both the refinement and the weight map predictions at once, and we can vary the upscaling factor to ensure that the network can adapt well to different amounts of upscaling.

##### IM-LUT

![IM LUT]({{ '/assets/images/01/imlut.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 5. An overview of the way in which the LUTs are created and used [3].*

Now, we have to transfer the IM-Net to LUTs. Given that IM-Net has 3 distinct sub-networks (scaling factor network, weight map network, and refinement network), we will have 3 LUTs; note that the refinement and weight map networks each use convolutional networks that leave each pixel of the output with a 2x2 receptive field in the input, as was seen with the original LUT. Additionally, each LUT is created with a sampling-interval scheme and interpolated indexing, as was also seen with the standard LUT (except for the scale LUT, which uses nearest neighbor interpolation). Specifically, the scale modulator LUT is indexed by a single number and outputs a vector of $$K$$ weights, making it small and efficient; the weight map LUT is indexed by 4 pixel values and outputs $$K$$ pixel values (since the weight map network *does not* perform any resizing), making it similar in size to the standard LUT; the refiner LUT is indexed by 4 pixels and outputs 1 pixel value (again, no resizing), making it smaller than the standard LUT. We see that, unlike the standard LUT, the size of the LUTs has no dependence on the scale factor; however, given that the refiner LUT will be used on the entire upscaled image, the overall computational complexity will increase with the scale factor (though, they note in the paper, it does not increase substantially with upscale factor).

Once the LUTs are computed, they can be used as stand-ins for the networks (as in the standard LUT), and then images can be upscaled as they were in IM-Net.

##### Results

![IM LUT qualitative]({{ '/assets/images/01/imlutqualitative.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 5. Qualitative results of IM-LUT compared to other methods [3].*

![IM LUT quantitative]({{ '/assets/images/01/imlutresults.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 5. Multiply-accumulate (MAC) operations, required storage, and PSNR for various super-resolution methods [3].*

Given the freedom to choose interpolation functions, the authors highlight a few possible combinations of functions: in the table above, N means "nearest neighbor", L means "bilinear", C means "bicubic" and Z means "lanczos", another kind of interpolation algorithm that uses windowed normalized sine functions; the letters, when put together, indicate that the given IM-LUT uses all of those interpolation functions. We see that the IM-LUT methods manage to be very efficient on storage and, for certain combinations of interpolations, on computations, as well; at the same time, they have much better performance than any single interpolation function, and, while not being state-of-the-art quantitatively, stay competitive with other LUT or similarly-efficient SR methods that require more storage and, in the case of the other LUT methods, cannot use a single model to adapt to different scaling factors.

![IM LUT weight maps]({{ '/assets/images/01/weightmaps.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 5. Analysis of weight maps for different interpolation functions [3].*

From the color-coded weight maps above, we see that the model is, in fact, using the weight maps to adapt to the input image. For instance, when nearest-neighbor and bilinear interpolation are used together, we see that the bilinear interpolation is prioritized in areas of low brightness, while the nearest neighbor method is used elsewhere. However, we also see that when nearest-neighbor, bilinear, and bicubic interpolation are used together, the nearest-neighbor interpolation has very little influence, indicating that one could even use the weight maps as a form of interpolation function selection, similar to how L1 regularization is used as variable selection in linear regression.

The IM-Net is already an interesting kind of reparameterization of the problem of learning-based super-resolution; instead of directly learning how to upscale an image, the network instead learns how to ensemble a set of given interpolation functions to do upscaling. This modular design allows for a great deal of flexibility in the model, as it can be combined with any arbitrary-scale upscaling method, even other learning-based methods, without any change to the core architecture. Of course, in this case, our real goal is a LUT-based method; in the realm of LUT-based methods, IM-LUT still stands out for its capability of arbitrary-scale super-resolution. Dynamically adapting to different scales will be a practical necessity for any super-resolution method, so it will be ineteresting to see if, as time goes on, efficient super-resolution methods will expand further in the direction of the IM-LUT, or if an entirely different architecture will rise to the top.

<!-- To achieve this fast runtime, a convolutional SR network is trained with a small receptive field, since the size of the SR-LUT grows exponentially with the receptive field size. This limitation introduces an inherent trade-off between PSNR and runtime: increasing the receptive field can improve reconstruction quality, but it also causes the LUT to expand dramatically, leading to slower performance.   

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


TODO: CIte IM-LUT -->


<!-- -adv: 
--made to be fast and small
-disadv: 
--practically, very small receptive field; lookup table grows exponentially(?) with RF size
--implicit assumptions about noise, blurring, downsampling of image based on training data -->

### Unfolding Networks

The "unfolding" in "unfolding network" refers to splitting up the problem of de-degredation into two distinct subproblems, that being (1) unblurring and upsampling, and (2) denoising. With this approach, one can show that problem (1) has a closed-form optimal solution that can explicitly adapt to specific given types of degradation with 0 learned parameters; this greatly reduces the burden on the learned portion of the network, which now only needs to do denoising. The method is a kind of fusing of model-based and learning-based approaches; despite involving a variant of a UNet, it is designed to be zero-shot adaptable to any kind of degradation that is parameterized by a known blurring kernel, downsampling factor, and noise level.

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

$$
\vec z_j  = \mathcal{F}^{-1} \left(\frac{1}{\alpha_j}\left(d - \overline{\mathcal{F}(j)} \odot_s \frac{(\mathcal{F}(j)d) \Downarrow_s}{(\overline{\mathcal{F}(j)}\mathcal{F}(j))\Downarrow_s + \alpha_j} \right) \right)
$$

where

$$

d = \overline{\mathcal{F}(j)}\mathcal{F}(\vec y \uparrow_s) + \alpha_j \mathcal{F}(x_{j-1})

$$

and $$ \mathcal{F}( \centerdot ) $$ denotes the discrete FFT and $$\odot_s$$ is the distinct block processing operator with element-wise multiplication.
$$ \Downarrow_s $$ denotes the block downsampler and $$ \uparrow_s $$ denotes the upsampler. The derivation of this solution is too long to include here, but is detailed in [4].

So, it remains to find 
$$\vec x_j = \arg \min_{\vec x} \Phi(\vec x) + \frac {\mu_j} {2\lambda} ||\vec x - \vec z_j||_2^2$$. 
One can notice that this is similar to our very first optimization target; indeed, finding $$\vec x_j$$ is equivalent to removing additive white Gaussian noise from $$\vec z_j$$ with $$\sigma^2_\text{noise} = \frac {\lambda} {\mu_j}$$ under a MAP framework. So, for convenience, we define 
$$\beta_j = \sqrt{\frac{\lambda}{\mu_j}}$$
 (the standard deviation of the Gaussian noise), and we have that finding $$x_j$$ is equivalent to a Gaussian denoising problem with noise level $$\beta_j$$ (or, $$\sigma^2_\text{noise} = \beta_j^2$$).

Given that this is a simple denoising task, we can opt for a denoising neural network. The paper in question uses a "ResUNet", which is a variant of a UNet:

![Unfolding UNet]({{ '/assets/images/01/UNet.png' | relative_url }})
{: style="max-width: 80%;"}
*Fig 1. UNet architecture as depicted in  [6].*

The network takes in $$ \vec z_k $$ and outputs the denoised image $$ \vec x_k $$. The ResUNet involves four scales, in which 2x2 strided convolutions and 2x2 transposed convolutions are adopted for down- and up-scaling, respectively, and residual blocks with the same structure as from ResNet are added during both downscaling and upscaling; of course, skip connections between downsampling and upsampling are also employed, as in the regular UNet.

In order to ensure adaptibility and nonblind-ness of the method, it is useful to incorporate the noise level $$\beta_j$$ into the network input. The paper's method for doing this is fairly simple: given an input image $$3 \times H \times W$$, a constant matrix with size $$H \times W$$ with all entries equal to $$\beta_j$$ is appended on the channel dimension to create an input of shape $$4 \times H \times W$$, which is fed into the network as normal. 


Now having created the strategy to accomplish the single image super resolution task, the paper makes the method explicit.
To do this they first introduce the Data module $$ \mathcal{D} $$, which is used to give us $$z_k = \mathcal{D}(\vec x_{k-1}, s, \vec k, \vec y, \alpha_k)$$, which is the optimal solution to

$$
\vec z_j = \arg \min_{\vec z} ||(\vec z \otimes \vec k)\downarrow_s - \vec y||_2^2 + \alpha_j ||\vec x_{j-1} - \vec z||_2^2 
$$

Of course, this is achieved with the closed form solution detailed above, so this module has no learnable parameters and can explicitly adapt to any blurring kernel, scale factor and noise level $$\sigma$$ (through $$\alpha_j$$).

The prior module $$ \mathcal{P} $$, which is used to give us $$x_k = \mathcal{P}(\vec z_k, \beta_j)$$, is trained to optimize the other objective

$$
\vec x_j = \arg \min_{\vec x} \lambda\Phi(\vec x) + \frac \mu 2 ||\vec x - \vec z_j||_2^2
$$

As alluded, this encapsulates the denoising ResUNet.

Lastly, the hyper-parameter module $$ \mathcal{H} $$ controls the outputs of the data and prior modules.
This module takes in $$ \sigma, s $$ and uses a MLP to predict optimal $$\lambda$$ and $$\mu_1, ..., \mu_K$$ for the given problem, which can then be used to compute $$\alpha_1, ..., \alpha_K$$ and $$\beta_1, ..., \beta_K$$.

With all of this, the authors note that the model is trained in a end-to-end fashion, where scaling factors, blurring kernels, and noise levels are chosen from a pool throughout training to generate LR samples. While certain portions of the network can explicitly adapt to these factors, the denoising network and hyperparameter module cannot do so as eaily, and so varying these values throughout training ensures that the network, as a whole, is robust and adaptable.

<!-- Putting all of the modules together, the Deep Unfolding Network is trained by first synthesizing low resolution
images from high resolution images, picking various scaling factors and Gaussian blur kernels. The L1 loss is used performance
since it promotes sharper images. Later in training, VGG perceptual loss and relativistic adversarial loss are included.
The model is optimized using batched gradient descent under the adam optimizer. -->

#### Results

![Unfolding Performance]({{ '/assets/images/01/dun_data.png' | relative_url }})
{: style="max-width: 100%;"}
*Fig 2. Average Peak Signal to Noise Ratio results of different methods, with the best results highlighted in red and second best in blue, as well as results of different image super resolution methods [2].*

As shown, the model from this paper outperforms its state-of-the-art peers (from the time).
Additionally, the qualitative results are favorable for the model; to note, the GAN variant of the model has generally sharper resolution, which can be attributed to the introduction of the more complex discriminator loss, compared to something like simple L1 loss.

In all, the deep unfolding network is an interesting blend of model-based and learning-based methods; they can both be very useful in their own right, and can empower each other when combined. The efficient, learning-free data module makes the job of the learned prior module much easier, and attains an optimal solution that could have required many times more parameters to replicate in a learning-based method. Conversely, the prior module can give us an accurate approximation for a complex, statistical problem, which model-based methods can struggle with.


<!-- -adv: 
--nonblind; explicitly adapts for diff blurring kernels, amts of noise, and downsampling factors
--limits learned parameters by only using learning for denoising step; deblurring and upsampling is closed-form, but still included in the chain of the model to allow learning to take advantage of it
-disadv: 
--sequential design w/ conv nets and (inverse) FFTs may be slow? 
--use of only conv nets for denoising step may limit effective receptive field

-more classical method can better adapt to diff blurring kernel, etc without extensive training; balance of learning based and model based method is good idk -->

## Experiments: An Extension of the Hybrid Attention Transformer

### Adaptive Sparse Transformer (AST)

While Transformers have shown immense success in vision tasks due to their ability to model complex, long-range relationships, this can also be a detriment; for reconstructing a given portion of an image, there are likely many parts of the image that are not relevant, and trying to use information from those parts will only add noise to your reconstruction. The self-attention mechanism of the regular Transformer has all tokens attend to all other tokens, with (Softmax) attention weights that will always be greater than 0. While the Transformer can use very small attention scores as a way of giving less attention to irrelevant tokens, it is likely useful to give the model more direct flexibility in selecting relevant image regions that does not require weights and scores of large magnitude; for this, [7] proposes the Adaptive Sparse Transformer (AST).

The AST changes the standard Transformer architecture in two ways: first, instead of just using Softmax on our attention scores to get attention weights, it introduces sparse score selection by using a weighted sum of the Softmax and squared-ReLU of the attention scores to get the attention weights, where the weight between the two schemes is a learnable parameter. With this, our Transformer has both "dense" (Softmax) attention weights, which allow all weights to be greater than 0, and "sparse" (squared ReLU) weights, where many weights will be exactly 0 and larger weights will be further amplified, and the model can learn how best to weigh these two methods for the task at hand. This self-attention mechanism is known as Adaptive Sparse Self-Attention (ASSA), and it is notable as a potentially powerful modification to the self-attention mechanism that requires just 1 extra parameter (as the weights can be computed as $$\text{sigmoid}(b), 1-\text{sigmoid}(b)$$ for $$b$$ learnable).

![AST Architecture]({{ '/assets/images/01/assa.png' | relative_url }})
{: style="max-width: 100%;"}
*Fig 1. Adaptive Sparse Self-Attention (ASSA)*

Secondly, the AST replaces the standard MLP of the Transformer with what the authors call a Feature Refinement Feed-forward Network (FRFN).

![AST Architecture]({{ '/assets/images/01/frfn.png' | relative_url }})
{: style="max-width: 100%;"}
*Fig 1. Feature Refinement Feed-forward Network (FRFN)*

Aside from being a post-attention network with convolutions that takes advantage of the spatial nature of the image, the FRFN aims to address redundancy in the channel dimension after using ASSA. It utilizes partial convolutions, which only convolves on a portion of the channels, and depthwise convolutions, which convolves each channel independently, along with splitting the image along its channels and then combining the two halves via matrix multiplication to efficiently provide a post-ASSA transformation that can further attenuate the information features extracted from the self-attention while giving the model some flexilibity to clear uninformative channels.

### Putting it Together

Recall the structure of the Hybrid Attention Transformer (HAT):

![HAT architecture]({{ '/assets/images/01/HAT_architecture.png' | relative_url }})
{: style="max-width: 90%;"}
*Fig 1. HAT Architecture Overview [1].*

One can see that the OCAB at the end of each RHAG acts as a sort of "gate" between blocks. Additionally, given the large key and value windows for the OCA, part of the purpose of the OCAB is to give each portion of the image a chance to incorporate together information from a larger portion of the image all at once, without relying on window-shifting. However, it is likely that portions of these larger windows are not useful for the image reconstruction task, so this larger window allows for ample noise to seep through into our image tokens. So, we propose a modification to the HAT architecture that replaces the attention mechanism of the OCA with the sparse attention of the AST, and, subsequently, replaces the MLP of the OCAB with the FRFN. We hypothesize that this can temper the OCA and prevent too many issues from the larger windows, but can also act as a overall information gate for each window itself at the end of the RHAG, as the windows will perform cross-attention where the set of key and value pixels are a superset of the query pixels.

Since we are giving the model this selective capability, we can likely afford to increase the size of the OCA window; in the original HAT paper, they had found that a overlap ratio of 0.5 was optimal (meaning that key and value windows were 1.5x as large as the query window in each dimension), but a value of 0.75 produced slightly worse results. So, we propose that increasing the overlap ratio to 0.75 will better utilize the sparse attention, giving the model access to more information that it can select from, while remaining computationally efficient.

Specifically, in our experiment, we will be modifying the architecture of the HAT-S, the smallest HAT model, for practical reasons, and training it for doing 4x image upscaling.

### Results

[results table here ]

Unfortunately, our modification to the HAT was unable to match the results of the regular HAT-S. A possible reason is that the sparse attention is not particularly useful for 4x upscaling; there are few pixels in the starting image compared to the output image, so each one is more likely to hold important information than if we were doing something like 2x upscaling. Additionally, the size of the overlap window may be an issue: we did not get the chance to experiment with other sizes, so it is possible that the smaller 0.5 ratio may have actually worked better for our model, still benefitting from the sparse attention mechanism.

[sample upscaled images from model ]

In any case, the sparse attention mechanism and FRFN are interesting modifications to the Transformer, and we hope that future research can come up with even more improvements on the architecture.

Our Colab training script for the experiment can be found here [LINK ], with supporting codebases here [LINK ] and here [LINK ].

## References

[1] X. Chen, X. Wang, J. Zhou, Y. Qiao and C. Dong, "Activating More Pixels in Image Super-Resolution Transformer," 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada, 2023, pp. 22367-22377, doi: 10.1109/CVPR52729.2023.02142.

[2] Zhang, Kai, et al. “Deep Unfolding Network for Image Super-Resolution.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 3217–3226.

[3] Jo, Younghyun, and Seon Joo Kim. “Practical Single-Image Super-Resolution Using Look-Up Table.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 691–700. doi:10.1109/CVPR46437.2021.00075.

[4] N. Zhao, Q. Wei, A. Basarab, N. Dobigeon, D. Kouamé and J. -Y. Tourneret, "Fast Single Image Super-Resolution Using a New Analytical Solution for ℓ2 – ℓ2 Problems," in IEEE Transactions on Image Processing, vol. 25, no. 8, pp. 3683-3697, Aug. 2016

[5] Park, S., Lee, S., Jin, K., & Jung, S.W. (2025). IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 14317-14325).

[6]  Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241.
Springer, 2015.

[7] S. Zhou, D. Chen, J. Pan, J. Shi and J. Yang, 
"Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration," 
2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2024, pp. 2952-2963, doi: 10.1109/CVPR52733.2024.00285. 

---
