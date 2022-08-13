# Butterflies Classification with Prototypical Network

### Quickstart

- Install the requirements using `pip install -r requirements.txt`
- place your photos to be predicted in the test examples folder
- run the following command
```bash
python predict.py -ex TestExamples --hybrid_model_weights checkpoints\best_model_95val_82tr.pth
```
- Results will be written into results.csv
- Classes Numbers correspond to the numbers in the [Butterflies200 Dataset](https://www.dropbox.com/sh/3p4x1oc5efknd69/AABwnyoH2EKi6H9Emcyd0pXCa?dl=0) in 
the "images_small" folder - 1

### Overview

This respiratory contains a solution for a common problem of class imbalance. 
The Butterfly200Dataset contains two main problems:

- It contains 200 classes with a number of examples per class that ranges from 30 samples to 800 samples.
- The single class contains male and female Butterflies each with underwing and upper wing views (many variants per class). 

- In such situations, a normal discriminative model can learn to perfectly classify the classes with big numbers of examples while not being able to
classify the classes with a small number of examples.

- On the other hand, generative models can learn to classify all the classes with less number of examples but they will have a hard time mapping the different variants in the same class to the same representation.

In this respiratory, a hybrid approach was used where a pre-trained ResNet was finetuned on a sub-task of classifying 4 classes with +400 examples each. 

This output of the first 2 layers from this ResNet was used to generate features for a prototypical network (ProtoNet) to classify the whole 200 classes with above 91% accuracy and at least 71% accuracy per class.


### Table of contents
1. [Usage](#usage)
2. [Results](#results)
3. [Approach](#approach)
4. [Credits](#credits)



<div style="text-align: center; padding: 10px">
    <img src="link" width="100%" style="max-width: 300px; margin: auto"/>
</div>


### Usage
To start download the [Butterflies200 Dataset](https://www.dropbox.com/sh/3p4x1oc5efknd69/AABwnyoH2EKi6H9Emcyd0pXCa?dl=0) and place it in a folder named 
"Data".

* For training the hybrid model using the pre-trained ResNet model
```bash
python train.py --freeze_encoder --enc_weights <weights path> --split_path <split dictionary path>  -lr 0.0001  --cuda
```
You can use "checkpoints\best_model_embed_res.pth" for the weights and "configs\splits\split_dict_hybrid_clust.pkl" for the split dictionary path.



* If you want to use the weights of a whole hybrid model as a start of the training you can use
```bash
python train.py --freeze_encoder --hybrid_model_weights <weights path> --split_path <split dictionary path> -lr 0.0001  --cuda
```

* After training and before the prediction, the means vectors of the classes must be calculated, to do that use
```bash
python construct_classes_means.py --hybrid_model_weights <weights path> --save_path <means saving path> --split_path <split dictionary path>  --no_precache
```
The training examples are used by default to construct the means vectors for the classes.


* To predict a set of examples place them in a folder and ensure they all have the same suffix, then use

```bash
python predict.py -ex <folder name> --hybrid_model_weights <weights path> --suffix *.jpg
```

#### Notes

A split dictionary is a dictionary with train, val, test keys with each having a list of all the corresponding samples paths.
You can refer to the notebook in "Notebooks\Splitting\TrainTestSplitHybrid.ipynb" for an example.

For faster training, the main preprocessing can be done using "Notebooks\DatasetPreperation\prepare_dataset.ipynp" to generate a ".pt" for loading (no augmentation will be performed in that case).

### Results

The model achieved 91% accuracy on the total dataset with 71% minimum accuracy per class.

To improve classes with low accuracy a hand split for these classes samples were done (can be found in "Notebooks\Splitting\TrainTestSplitHybrid.ipynb").


### Approach

- A pre-trained ResNet is finetuned on classes with more than 400 samples as a sub-classification task.

- Then a hybrid model is made of the first two layers of the ResNet with a Protonet. 

- The Protonet is trained using the features from the ResNet while the ResNet weights were freezed

- The whole model was finetuned with a low learning rate to get the best results

The model was performing badly on certain classes (less than 60% accuracy) which degraded the minimum classes' accuracy and the whole accuracy. 

This was attributed to big differences between train-val-test splits as the number of examples was low.

Splits for some of these classes were hand-made to assure having reasonable splits.
This data-centric method was able to get these classes from about 50% accuracy to more than 95% accuracy.

This way takes a lot of time but it is very effective. There are still classes that need to be dealt with the same way if you want to increase the performance.

### Credits

The Prototypical  Network implementation is used from [Prototypical-Networks-for-Few-shot-Learning-PyTorch](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)

