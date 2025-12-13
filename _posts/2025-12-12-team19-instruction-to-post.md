---
layout: post
comments: true
title: From Paris to Seychelles - Deep Learning Techniques for Global Image Geolocation
author: Shelby Falde, Joshua Li, Alexander Chen
date: 2025-12-12
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

Image geolocation is the task of predicting the geographic coordinates (latitude and longitude) of a given image. While humans rely on specific landmarks or cultural context, training a machine to recognize any location on Earth is a massive classification and regression challenge.

Early approaches relied on matching images to databases of landmarks, but these failed in "in the wild" scenarios where no distinct landmark is visible (e.g., a random road in rural Norway). Deep learning has since shifted this paradigm by learning global feature distributions and characteristics around the world.

In this survey, we explore how architectures have evolved from Convolutional Neural Networks (CNNs) to Transformers and CLIP-based models, specifically focusing on how they handle the partitioning of the Earth and the loss functions used to train them.

## PlaNet: The Classification Approach (2016)

The pioneering paper *Photo Geolocation with Convolutional Neural Networks* [1], known as *PlaNet*, framed geolocation not as a regression problem (predicting raw coordinates), but as a classification problem.

### The Concept of Geocells
To treat the world as a classification target, PlaNet subdivides the Earth into discrete regions called *Geocells*.

However, a uniform grid doesn't work well because photo distribution is not uniform since there are millions of photos of Paris, but very few of the middle of the ocean for example. PlaNet thus introduces *adaptive partitioning*:

1. Start with an S2 geometry grid.
2. Recursively subdivide cells that contain too many photos.
3. Stop when a cell reaches a target photo count.

This results in a map where dense urban areas have tiny, precise cells, while oceans and deserts have massive cells.

![PlaNet Geocells]({{ '/assets/images/team19/planet_geocells.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Adaptive partitioning of the world into Geocells based on photo density [1].*

### Architecture and Sequence Learning
PlaNet utilizes an *Inception V3* architecture to classify images into one of these 26,263 geocells.

A key innovation in PlaNet was the use of *LSTMs (Long Short-Term Memory)* networks. Geolocation often happens in the context of a photo album. Knowing that the previous photo was taken at the Eiffel Tower heavily implies the current photo of a generic croissant is likely in Paris.

The model then outputs a probability distribution

$$
P(c_i | I)
$$

over all geocells using the features extracted by the CNN backbone.

## PIGEON: The Semantic Shift (2023)

The next leap forward comes from *PIGEON* (Pre-trained Image GEO-localization Network) [2]. This model was designed to compete against top human *GeoGuessr* players. Geoguessr is a popular browser game designed around humans guessing where you are in the world based on an image or 3D explorable space travelable via Google Streetview.

![Pigeon Geocells]({{ '/assets/images/team19/pigeon_diagram.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Prediction pipeline and main contributions of PIGEON.*

### Semantic Geocells
PIGEON refines the geocell concept. While PlaNet split cells based purely on photo density, PIGEON incorporates *administrative boundaries*.

*   *PlaNet:* Splits a cell if it has too many photos, regardless of borders.
*   *PIGEON:* Tries to respect country/region borders. This creates "Semantic Geocells," preventing the model from confusing two neighboring countries that might look similar but have different road markings or signage.

![Pigeon Diagram]({{ '/assets/images/team19/naive_semantic_geocells.png' | relative_url }}) 
{: style="width: 800px; max-width: 100%;"}
*Fig 3. &emsp;&emsp;&emsp;&emsp; a) Old Naive geocells &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; b) Pigeon’s semantic geocells [2].*

### Haversine Smoothing Loss
A major limitation of PlaNet's classification approach is that it penalizes "near misses" just as harshly as "far misses." If the model guesses a cell 1km away from the correct one, standard One-Hot encoding treats it the same as guessing a cell on a different continent.

To fix this, PIGEON replaces standard labels with *Haversine Smoothing*. Instead of the correct cell being a `1` and all others `0`, neighboring cells get a partial label based on their physical distance to the image.

First, they define the **Haversine Distance** between two points $$\mathbf{p}_1$$ and $$\mathbf{p}_2$$ on Earth:

$$
\text{Hav}(\mathbf{p}_1, \mathbf{p}_2) = 2r \arcsin \left( \sqrt{\sin^2 \left( \frac{\phi_2 - \phi_1}{2} \right) + \cos(\phi_1) \cos(\phi_2) \sin^2 \left( \frac{\lambda_2 - \lambda_1}{2} \right)} \right)
$$

They then generate a "smoothed" label $$y_{n,i}$$ for every geocell $$i$$ relative to the true image location $$\mathbf{x}_n$$. Cells closer to the true location receive a higher value:

$$
y_{n,i} = \exp \left( - \frac{\text{Hav}(\mathbf{g}_i, \mathbf{x}_n) - \text{Hav}(\mathbf{g}_n, \mathbf{x}_n)}{\tau} \right)
$$

Finally, the Loss Function $$\mathcal{L}_n$$ minimizes the difference between the predicted probability $$p_{n,i}$$ and this smoothed distance label:

$$
\mathcal{L}_n = - \sum_{g_i \in G} \log(p_{n,i}) \cdot y_{n,i}
$$

This ensures that the gradient penalty is lower if the model predicts a location that is geographically close to the target, effectively teaching the model a continuous topology of the Earth.

### CLIP Pre-training & Retrieval
PIGEON utilizes *CLIP (Contrastive Language-Image Pre-training)* from OpenAI as its backbone. CLIP is trained on 400 million image-text pairs, so it already possesses a deep understanding of visual concepts, even before it is fine-tuned for location. It also introduces *intra-cell retrieval* to refine the exact location by looking up similar images within the predicted geocell.

### Results
After several other finetuning features discussed in the paper but not here, *PIGEON* achieved landmark results and near pefect accuracy on country and continent classification:

![Pigeon Diagram]({{ '/assets/images/team19/pigeon_results.png' | relative_url }}) 
{: style="width: 800px; max-width: 100%;"}
*Fig 4. PIGEON Model results on a holdout dataset of 5,000 Street View locations.*

While some may call the above results unimpressive, they then compare PIGEON's results to ranked players on GeoGuessr to more fairly contextualize results:
![GeoGuessr Context]({{ '/assets/images/team19/geoguessr_context.png' | relative_url }}) 
{: style="width: 800px; max-width: 100%;"}
*Fig 4. PIGEON Model Comparison to Ranked GeoGuessr players. Champion Division being top 0.01% of all players.*

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

## References

[1] Weyand, Tobias, Ilya Kostrikov, and James Philbin. "Planet-photo geolocation with convolutional neural networks." *European Conference on Computer Vision*. Springer, Cham, 2016.

[2] Haas, Lukas, et al. "PIGEON: Predicting Image Geolocations." *arXiv preprint* arXiv:2307.05845, 2023. https://doi.org/10.48550/arXiv.2307.05845 (Accepted at CVPR 2024.)
