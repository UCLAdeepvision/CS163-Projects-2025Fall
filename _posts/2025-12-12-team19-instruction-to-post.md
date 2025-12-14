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

## Translocator: A New Architecture

The next major advancement in tackling geolocation as a classification problem is introduced in the paper *Where in the World is this Image?
Transformer-based Geo-localization in the Wild*. The paper proposes TransLocator, a fundamentally different model architecture to PlaNet that makes use of transformers and semantic segmentation maps [2].

### Transformers + Segmentaion

The Translocator uses a vision transformer as the backbone of the model, and contains two parallel vision transformer branches. One branch's input is the RGB image, and the other's is a semantic segmentation map of the image, obtained from  HRNet pretrained on ADE20K.

![Translocator]({{ '/assets/images/team19/translocator.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Overview of the proposed model TransLocator* [2].

## PIGEON: The Semantic Shift (2023)

The next leap forward comes from *PIGEON* (Pre-trained Image GEO-localization Network) [3]. This model was designed to compete against top human *GeoGuessr* players. Geoguessr is a popular browser game designed around humans guessing where you are in the world based on an image or 3D explorable space travelable via Google Streetview.

![Pigeon Geocells]({{ '/assets/images/team19/pigeon_diagram.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Prediction pipeline and main contributions of PIGEON.*

### Semantic Geocells
PIGEON refines the geocell concept. While PlaNet split cells based purely on photo density, PIGEON incorporates *administrative boundaries*.

*   *PlaNet:* Splits a cell if it has too many photos, regardless of borders.
*   *PIGEON:* Tries to respect country/region borders. This creates "Semantic Geocells," preventing the model from confusing two neighboring countries that might look similar but have different road markings or signage.

![Pigeon Diagram]({{ '/assets/images/team19/naive_semantic_geocells.png' | relative_url }}) 
{: style="width: 800px; max-width: 100%;"}
*Fig 3. &emsp;&emsp;&emsp;&emsp; a) Old Naive geocells &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; b) Pigeon’s semantic geocells [3].*

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

## ETHAN


ETHAN introduces a framework that leverages existing large vision language models(LVLMs). While PlaNet and PIGEON achieved impressive results, they operate as pattern matching systems, relying on visual features tied to geocells. ETHAN, instead, attempts to mimic human geoguessing strategies by analyzing visual and contextual cues such as architectural, natural, and cultural elements. It replicates this reasoning process through chain-of-thinking prompting built on top of existing VLMs.


### Vision-Language Models:


ETHAN is a prompting framework that relies on existing VLM models. They test their prompting framework on top of GPT-4o and LLaVA. These models have the architecture of Vision Encoder + projection + generative language model, combining context from image and text tokens. Ethan leverages this pre-existing knowledge without additional training




### Chain-Of-Thought Geolocation


The core innovation lies in the use of chain-of-thought(CoT) prompting that guide VLMs through structured geolocation reasoning. ETHAN produces intermediate reasoning steps that mirror human geographic deduction.


A typical reasoning hierarchy looks like the following:


1. Visual clue extraction: architecture, vegetation, road signs, infrastructure
2. Geographic constraint Reasoning: apply world knowledge to narrow possibilities
3. Progressive Refinement: Continent->Country->Region->Coordinates
4. Uncertainty Quantification: Express confidence and alternate hypotheses


### GPS Module Prompting Strategy:


ETHAN introduces the GPS (Geolocation Prompting Strategy) module, which structures how queries are presented to VLMs. Since VLMs are general-purpose models not specifically trained for geolocation, careful prompt design is crucial for strong geographic reasoning.


A typical prompt looks like:
>You are the leading expert in geolocation research. You
have been presented with an image, and your task is to
determine its precise geolocation, specifically identifying
the country it was taken in. To accomplish this, examine the
image for broad geographic indicators such as architectural
styles, natural landscapes, language on signs, and culturally
distinctive elements to suggest a particular country. Narrow
down the location by identifying regional characteristics
like specific flora and fauna, types of vehicles, and road
signs that can indicate a particular region or subdivision
within the country. Focus on highly specific details within
the image, such as unique landmarks, street names, or
business names, to pinpoint an exact location. For instance,
if the place is address, with coordinates lat, lon, explain
how these elements led you to this conclusion by analyzing
visual clues, cross-referencing data with known geographic
information, and validating your findings with additional
sources


The GPS module can be enhanced with few-shot examples, providing 2-3 sample images with reasoning chains to prime the VLM's logical patterns and calibrate confidence levels.

### Deriving Coordinates

To calculate coordinates form textual reasoning, ETHAN employs several strategies:
1. Direct Coordinate Prediction: VLMs output coordinates for recognized landmarks
2. Geocoding Integration: Convert place names ("Stockholm, Sweden") to coordinates via APIs
3. Hierarchical Averaging: Use region centroids when only broad areas are identified
4. Multi-Hypothesis Handling: Process multiple candidates with associated probabilities


### Results

![Ethan Results]({{ '/assets/images/team19/ethan_results.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 5. Results of ETHAN Prompting framework with SOTA LVLMs on custom dataset[4].*


ETHAN performs strongly, with high accuracy in country and continent classification, on par with Pigeon. As compared to previous strategies, ETHAN benefits from increased interpretability, zero-shot generalization, but has higher computational costs as VLM inference is slower. Additionally, it can suffer from hallucination risks where generative model recognizes non-existent models or applies incorrect assumptions.

## Conclusion

The progression from PlaNet to ETHAN illustrates a fundamental shift in image geolocation architectures. Early CNN-based approaches like PlaNet established geocell classification as a viable framework, while TransLocator demonstrated the advantages of transformer architectures and multi-modal inputs through semantic segmentation. 

Future work will likely focus on hybrid architectures that combine the computational efficiency of learned embeddings with the reasoning capabilities of VLMs. Possible research directions include developing loss functions that better capture hierarchical geographic relationships, improving geocell partitioning strategies that balance semantic coherence with training efficiency, and addressing the persistent challenge of performance degradation in underrepresented geographic regions where training data remains sparse.


## References

[1] Weyand, Tobias, Ilya Kostrikov, and James Philbin. "Planet-photo geolocation with convolutional neural networks." *European Conference on Computer Vision*. Springer, Cham, 2016.

[2] Pramanick, Shraman, et al. "Where in the World is this Image? Transformer-based Geo-localization in the Wild." 	arXiv:2204.13861, 2022. https://doi.org/10.48550/arXiv.2204.13861

[3] Haas, Lukas, et al. "PIGEON: Predicting Image Geolocations." *arXiv preprint* arXiv:2307.05845, 2023. https://doi.org/10.48550/arXiv.2307.05845 (Accepted at CVPR 2024.)

[4] Liu, Yi, et al. "Image-Based Geolocation Using Large Vision-Language Models." 	arXiv:2408.09474, 2024. 
https://doi.org/10.48550/arXiv.2408.09474

