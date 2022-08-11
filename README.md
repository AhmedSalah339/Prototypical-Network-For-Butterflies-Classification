# Title

### Quickstart

Install with `pip install x`
```python
import numpy as np
```

Or find a Google Colab example [here](https://colab.research.google.com/).  

### Overview
This repository contains a PyTorch implementation .

The goal of this implementation is to be x, y, and z. 

At the moment, you can easily:
 * x
 * y
 * z



### Table of contents
1. [Title1 X](#title1-x)
2. [Title2 Y](#title2-y)
3. [Title3 z](#Title3-z)
4. [Usage](#usage)
    * [Train](#train)
    * [Example: Classify](#example-classification)
6. [Contributing](#contributing)

### About ViT

Visual Transformers (ViT) are a straightforward application of the [transformer architecture](https://arxiv.org/abs/1706.03762) to image classification. Even in computer vision, it seems, attention is all you need. 

The ViT architecture works as follows: (1) it considers an image as a 1-dimensional sequence of patches, (2) it prepends a classification token to the sequence, (3) it passes these patches through a transformer encoder (like [BERT](https://arxiv.org/abs/1810.04805)), (4) it passes the first token of the output of the transformer through a small MLP to obtain the classification logits. 
ViT is trained on a large-scale dataset (ImageNet-21k) with a huge amount of compute. 

<div style="text-align: center; padding: 10px">
    <img src="https://raw.githubusercontent.com/google-research/vision_transformer/master/figure1.png" width="100%" style="max-width: 300px; margin: auto"/>
</div>


### Title1 X



### Title2 Y

Install with pip:
```bash
pip install x
```



### Usage

#### Table


Details:

|    *Name*         |     *Models*    |  *Results*   |*Available? *|
|:-----------------:|:---------------:|:------------:|:-----------:|
| `n1`              |  im1            | - nm         |      ✓      |
| `n2`              |  im2            | - mn         |      ✓      |


#### Custom ViT

Loading custom configurations is just as easy: 
```python
from pytorch_pretrained_vit import ViT
# The following is equivalent to ViT('B_16')
config = dict(hidden_size=512, num_heads=8, num_layers=6)
model = ViT.from_config(config)
```

#### Example: Classification

<!-- TODO: new Colab -->

Pin Points: 
 - Pin1
 - Pin2

### Contributing


