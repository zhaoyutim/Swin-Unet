import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape[0], image.shape[1]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class PalsarDataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir
        # self.train_dataset = np.load('/geoinfo_vol1/zhao2/proj2_dataset/proj2_train.npy')
        self.train_dataset = np.load('/Users/zhaoyu/PycharmProjects/ee_fire_monitoring/dataset/proj2_test.npy')
        self.y_dataset = self.train_dataset[:,:,:,3]>0
        self.image, self.image_val, self.label, self.label_val = train_test_split(self.train_dataset[:,:,:,:3], self.y_dataset, test_size=0.2, random_state=0)

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        sample = {'image': self.image[idx,:,:,:], 'label': self.label[idx,:,:]}
        if self.transform:
            sample = self.transform(sample)
        return sample
