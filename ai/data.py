import os
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from constants import class_labels, IMAGE_SIZE

train_dir = "./seg_train"
test_dir = "./seg_test"

data_transform = transforms.Compose(
    [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]
)


class MyDataset(Dataset):
    def __init__(self, folder: str, transform=data_transform):
        self.images_folder = folder
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


def get_dataloader(
    mode: Literal["train", "test"], *, batch_size: int = 32, shuffle: bool = True
) -> DataLoader:
    dataset = MyDataset(train_dir if mode == "train" else test_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
