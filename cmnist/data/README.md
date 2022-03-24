# MNIST datasets

The MNIST main dataset and split MNIST datasets will be saved here.
Use the `create_split_mnist` function to create the splits. Example use:

    from torchvision.datasets import MNIST
    from split_mnist import *
    
    MNIST("data/", download=True)  # download the main MNIST dataset

    create_split_mnist("data/", [0,1], save=True)
    create_split_mnist("data/", [2,3], save=True)

creates a split containing classes 0 and 1; and a split containing classes 2 and 3.