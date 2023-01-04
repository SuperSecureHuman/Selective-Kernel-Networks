# Sknets

## Other papers

<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7680558/> - 98 to 99% No lung opacity

<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8958819/> - ML models - 90+ - No lung opacity

<https://ieeexplore.ieee.org/document/9344870> - 3 class, 96  

<https://www.hindawi.com/journals/jhe/2021/6799202/> - no lung opacity

## Lung Opacity class, and why it is important

Lung Opacity (Non-COVID lung infection)
We can consider this class as a misc class, which dosent inculde other other diseases. But we have included this class because, we wanted to classify even an infenction. If this class was ignored, then there is a good chance that these images might be classified as normal

## Data

To preserve the class balance - we have taken around 1300 to 1400 images in every class - With the following augmentations

RandomHorizontalFlip(p=0.5)
RandomRotation(degrees=30)
RandomVerticalFlip(p=0.5)

Image input size is 224x224

Normalizaion - In 3 channel mode, we have used the following normalization

70% train, 20% val, 10% test

## Selective Kernel Networks

* tell about sknet first (with the paper) add stuff like receptive vision
* tell about the skconv block
* cool graph of the model
* parameter table with number of params (in our case)
* tell about the highway/skip connections being present in sknet (which is also present in resnet)
* tell about the 3x3 and 5x5 kernels being used
* tell about dilation

## Models and results

Here, we present a new model - Selective kernel networks.

We have compared the performance of Sknet and Resnet, and did a study on convergence and overfitting pattern - With and without augmentation.

We used TIMM models for premade resnet models. Due the fact that timm models support only 3 dim input, we have resorted to 3 dim input for our sknet blocks.

When there is no augmentation, we can see that sknet kinda avoids overfit.

And in general, sknet converges faster than resnet.

Since in general, the accuracy is around 90 to 93%, we can provide inference at what epoch the model got that val accuracy. With both the learning rates.

Table with the following

* Model
* Train loss
* Val loss
* Train acc
* Val acc
* Test loss @ best val
* Test acc @ best val
* F1
* Precision
* Recall
* Num Params of the model

## Graphs

IDK how to effectively show stuff here...

Scatter plot of test set

Make sure to have y-axis from 50 to 100 (makes graphs cleaner)
