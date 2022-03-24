import os
import torch
from torchvision.datasets import MNIST
from typing import List


class SplitMNIST(MNIST):
    def __init__(self, root: str, classes: List[int], **kwargs):
        self.classes = classes
        try:
            super().__init__(root, **kwargs)
        except RuntimeError:
            raise RuntimeError("Dataset loading failed, maybe it does not exist. " +
                               "Use create_split_mnist(root, [classes]) to create one.")

    @property
    def raw_folder(self):
        raise NotImplementedError("Split MNIST does not have raw data")

    @property
    def processed_folder(self):
        return os.path.join(self.root,
                            f'{self.__class__.__name__}{"".join(str(i) for i in sorted(self.classes))}',
                            'processed')


def create_split_mnist(root: str, classes: List[int], save=False):
    for is_training, savename in [(True, "training.pt"), (False, "test.pt")]:
        orig_mnist = MNIST(root, train=is_training)
        imgs = orig_mnist.data
        tgts = orig_mnist.targets

        mask = sum(tgts == cls for cls in classes)  # make a bool tensor, the sum behaves like an 'or'
        expanded_mask = mask.view(-1, 1, 1).expand(-1, 28, 28)  # expand the mask to match the 28x28 img shape
        filtered_imgs = torch.masked_select(imgs, expanded_mask).view(-1, 28, 28)
        filtered_tgts = torch.masked_select(tgts, mask)

        if save:
            savedir = os.path.join(root, f'SplitMNIST{"".join(str(i) for i in sorted(classes))}/processed')
            if not os.path.exists(savedir):
                os.makedirs(savedir, exist_ok=True)
            with open(os.path.join(savedir, savename), mode="wb") as f:
                torch.save((filtered_imgs, filtered_tgts), f)
        return filtered_imgs, filtered_tgts
