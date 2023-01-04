# Points maybe?

Compare the convergence pattern of resnet and sknet - There is a good difference

SkNet dosent over fit like resnet - This could mean that it generalizes better

We are using 4 classes instad of 3 - A better summary of lung opacity class is needed

We have used 3x3 with dil 1 and 3x3 with dil 2 kernels, for effective kernel sizes of 3x3 and 5x5

Trying to compare attenetion like the original paper could be done, but not sure how to do it like the author did (maybe for journal)

Provide num of params and flops for each model (easy loop with thop)

Mention about the augmentations used

    ```python
    torchvision.transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomRotation(degrees=30),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    ```

Gib insighs on with and without augmentations

Provide model graph for skConv block

Make sure to give the insight that highway/skip connections are used here

## Cites till now

<https://paperswithcode.com/paper/selective-kernel-networks>

<https://arxiv.org/abs/1903.06586>

<https://github.com/implus/SKNet>

<https://github.com/developer0hye/SKNet-PyTorch/blob/master/sknet.py>

"We used Weights & Biases for experiment tracking and visualizations to develop insights for this paper."  # From wandb site

Tensorboard (?)

pytorch citation

<https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database>
