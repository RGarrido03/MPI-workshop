import os
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset

from constants import class_labels

train_dir = "./seg_train"
test_dir = "./seg_test"


class MyDataset(Dataset):
    def __init__(self, type: Literal["train", "test"], transform=None):
        self.images_folder = train_dir if type == "train" else test_dir
        self.transform = transform

        self.df = [
            (os.path.join(self.images_folder, folder, file), class_labels[folder])
            for folder in os.listdir(self.images_folder)
            if folder in class_labels
            for file in os.listdir(os.path.join(self.images_folder, folder))
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        filename, label = self.df[index]
        image = Image.open(filename)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
