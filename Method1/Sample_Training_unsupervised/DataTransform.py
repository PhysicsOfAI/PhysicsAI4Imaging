import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, random_split
import torch


class ImageTransform(Dataset):

    def __init__(self, csv_file, dir, transform=None):
        """ Forward operator for real images
        :param csv_file: path to the csv file containing all good image file names
        :param root_dir:  directory with all the images
        :param transform: optional transform to be applied on a sample
        :return: transformed image
        """
        self.image_list = pd.read_csv(csv_file, header=None, index_col=False, squeeze=True)
        self.dir = dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        baseDir, tailDir = os.path.split(self.image_list.iloc[idx])
        file_name = tailDir

        path = os.path.join(self.dir, baseDir, file_name) #creating the full path to the first image
        path = os.path.abspath(path) #creating the full path to the first image

        img = io.imread(path)

        if self.transform:
            img = self.transform(img)

        img = img.view(-1)

        return img


def split(dataset, ratio_train=0.8):

    """ split dataset in training and validation data
    :param dataset: all data
    :param ratio_train:  ratio of training data
    :return: training and validation set
    """
    # set random seed
    torch.manual_seed(42)

    train_len = int(len(dataset) * ratio_train)
    split_data = random_split(dataset, [train_len, len(dataset) - train_len])
    torch.manual_seed(torch.initial_seed())

    return split_data

