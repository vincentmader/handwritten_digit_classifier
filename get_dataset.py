import torch
from torchvision import transforms, datasets

from config import PATH_TO_TRAINING_DATA


def main():

    train = datasets.MNIST(  # set of 28x28 images of handwritten digits
        PATH_TO_TRAINING_DATA, train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    test = datasets.MNIST(
        PATH_TO_TRAINING_DATA, train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    trainset = torch.utils.data.DataLoader(
        train, batch_size=10, shuffle=True  # shuffle -> generalization!
    )  # batch size 8-64 -> helps generalizing & run time!
    testset = torch.utils.data.DataLoader(
        test, batch_size=10, shuffle=True
    )

    return trainset, testset
