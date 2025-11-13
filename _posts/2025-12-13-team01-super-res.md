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

base LUT is very simple, may be good to highlight one of the many recent extensions to it

-adv: 
--made to be fast and small
-disadv: 
--practically, very small receptive field; lookup table grows exponentially(?) with RF size
--implicit assumptions about noise, blurring, downsampling of image based on training data

### Unfolding Methods

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



---
