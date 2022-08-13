# Butterflies Classification with Prototypical Network

### Quickstart

- Install the requirements using `pip install -r requirements.txt`
- place your photos to be predicted in the test examples folder
- run the following command
```bash
python predict.py -ex TestExamples --hybrid_model_weights checkpoints\best_model_95val_82tr.pth
```
- Results are will be written into results.csv
- Classes Numbers corresponds to the numbers in the [Butterflies200 Dataset](https://www.dropbox.com/sh/3p4x1oc5efknd69/AABwnyoH2EKi6H9Emcyd0pXCa?dl=0) in 
the "images_small" folder

### Overview

This respiratory contains a solution for a common problem with classes impalance. 
The Butterfly200Dataset contains two main problems:

- It contains 200 classes with number of examples per class that ranges from 30 samples to 800 samples.
- The single class contains male and female Butterflies each with underwing and upperwing views (many variants per class). 

In such situations a normal discriminative model can learn to perfectly classify the classes with big numbers of examples while not being able to
classify the calsses with small number of examples.

On the other hand, generative models can learn to classify all the classes with less number of examples but it will have a hard time mapping the different variants in the same class to the same representation.

In this respiratory a hybrid approach was used where a pretrained ResNet was finetuned on a sub-task of classifying 4 classes with +400 examples each. 

This output of the first 2 layers from this ResNet was used to generate features for a prototypical network (ProtoNet) to classify the whole 200 classes with above 91% accuracy and at least 71% accuracy per class.


### Table of contents
1. [Usage](#usage)
2. [Results](#results)
3. [Dataset Overview](#dataset-overview)
4. [Approach](#approach)
    * [Table](#table)
    * [Pin Points](#pin-points)



<div style="text-align: center; padding: 10px">
    <img src="link" width="100%" style="max-width: 300px; margin: auto"/>
</div>


### Usage
To start download the [Butterflies200 Dataset](https://www.dropbox.com/sh/3p4x1oc5efknd69/AABwnyoH2EKi6H9Emcyd0pXCa?dl=0) and place it in a folder named 
"Data".

For training the hybrid model using the pretrained ResNet model
```bash
python train.py --freeze_encoder --enc_weights <weights path> --split_path <split dictionary path>  -lr 0.0001  --cuda
```
This freezes the ResNet encoder and uses the weights of the ResNet trained on the sub-task.

You can use "checkpoints\best_model_embed_res.pth" for the weights and "configs\splits\split_dict_hybrid_clust.pkl" for the split dictionary vth.



If you want to use the weights of the best model as a start of the training you can use
```bash
python train.py --freeze_encoder --hybrid_model_weights <weights path> --split_path <split dictionary path> -lr 0.0001  --cuda
```
You can use "checkpoints\best_model_95val_82tr.pth" for the weights.

After training and before the prediction the means vectors of the classes msust be calculated, to do that use
```bash
python construct_classes_means.py --hybrid_model_weights <weights path> --save_path <means saving path> --split_path <split dictionary path>  --no_precache
```
The training examples are used by default to construct the means vectors for the classes.

To predict a set of examples place them in a folder and insure they all has the same suffix, then use

```bash
python predict.py -ex <folder name> --hybrid_model_weights <weights path> --suffix *.jpg
```

### Results

Install with pip:
```bash
pip install x
```
### Citation


### Pinpoints

#### Table


Details:

|    *Name*         |     *Models*    |  *Results*   |*Available? *|
|:-----------------:|:---------------:|:------------:|:-----------:|
| `n1`              |  im1            | - nm         |      ✓      |
| `n2`              |  im2            | - mn         |      ✓      |



#### Pin Points

<!-- TODO: new Colab -->

Pin Points: 
 - Pin1
 - Pin2



