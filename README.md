# Selective Kernel Networks for Chest X-Ray Classification

This repository contains the code to evaluate the performance of the Selective Kernel Networks (SKN) on the Chest X-Ray dataset. 

## Requirements

The code is written in Python 3.6. The required packages are listed in the requirements.txt file.

## Data

<https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database>

## Usage

timm_train.py deals with the training of the model using the TIMM library. This was used to get the final results.

Sample code to run the code:

```bash

python timm_train.py --learning_rate=0.00005 --epochs=30 --batch_size=32 --model="resnet18" --wandb=True --data="/Covid_Radiography_Project/Dataset/Balanced" --seed=10 --image_size=224
```