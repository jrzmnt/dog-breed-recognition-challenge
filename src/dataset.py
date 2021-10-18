# regular imports
import os
import re
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import logging
import pickle
from PIL import Image
from tqdm import tqdm


# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# pytorch related imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable

# lightning related imports
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer


class DogDataset(Dataset):

    def __init__(self, file_list, class_dict, transform=None, phase=None):
        self.file_list = file_list
        self.class_dict = class_dict
        self.transform = transform
        self.phase = phase
        print(file_list[0])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')

        # Transformimg Image
        img_transformed = self.transform(img, self.phase)

        # Get Label
        label = img_path.split('\\')[0].split('-')[-1]
        label_id = self.class_dict[label]

        return img_transformed, label_id


class ImageTransform():

    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.data_transform = {

            'train': transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(degrees=(0, 180)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            'train_enroll': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        }

    def __call__(self, img, phase):

        return self.data_transform[phase](img)


class DogDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './../data/dogs/'):
        super().__init__()

        # images and dataset
        self.data_dir = data_dir
        self.train_data_dir = data_dir + 'train/'
        self.test_data_dir = data_dir + 'recognition/test/'
        self.enroll_data_dir = data_dir + 'recognition/enroll/'

        self.batch_size = batch_size
        self.dims = (3, 224, 224)
        self.img_size = 224
        self.num_classes = 100
        self.class_dict = {}
        self.enroll_class_dict = {}

        self.train_path_images = []
        self.test_path_images = []
        self.enroll_path_images = []
        self.breeds = []
        self.enroll_breeds = []

        self.train_split_path = []
        self.val_split_path = []
        self.dog_train = []
        self.dog_val = []
        self.dog_test = []
        self.dog_enroll = []

    def prepare_data(self):
        """Scan train_data_dir and test_data_dir to save all the train/test paths.
           It also creates train_dict to map the classes available to a scalar.
        """

        # Scan all train/val images
        for d in os.listdir(self.train_data_dir):
            dog_dir = os.path.join(self.train_data_dir, d)

            for img in os.listdir(dog_dir):
                img_path = os.path.join(dog_dir, img)
                self.train_path_images.append(img_path)
                self.breeds.append(img_path.split('\\')[0].split('-')[-1])

        self.breeds = set(self.breeds)

        for idx, breed in enumerate(self.breeds):
            self.class_dict[breed] = idx

        # Scan all test images
        for d in os.listdir(self.test_data_dir):
            dog_dir = os.path.join(self.test_data_dir, d)

            for img in os.listdir(dog_dir):
                img_path = os.path.join(dog_dir, img)
                self.test_path_images.append(img_path)
                self.enroll_breeds.append(
                    img_path.split('\\')[0].split('-')[-1])

        self.enroll_breeds = set(self.enroll_breeds)

        for idx, enroll_breed in enumerate(self.enroll_breeds):
            self.enroll_class_dict[enroll_breed] = idx

        print(50*'-')
        print(f'Images available for train/val: {len(self.train_path_images)}')
        print(f'Classes available for train/val: {len(self.breeds)}')
        print()
        print(f'Images available for test: {len(self.test_path_images)}')
        print(50*'-')

    def prepare_enroll_data(self, ):
        """Scan enroll_data_dir to save all the enroll paths.
           It also creates enroll_dict to map the classes available to a scalar.
        """

        # Scan all enroll images
        for d in os.listdir(self.enroll_data_dir):
            dog_dir = os.path.join(self.enroll_data_dir, d)

            for img in os.listdir(dog_dir):
                img_path = os.path.join(dog_dir, img)
                self.enroll_path_images.append(img_path)
                self.enroll_breeds.append(
                    img_path.split('\\')[0].split('-')[-1])

        self.enroll_breeds = set(self.enroll_breeds)

        for idx, enroll_breed in enumerate(self.enroll_breeds):
            self.enroll_class_dict[enroll_breed] = idx

    def setup(self, stage=None, enroll=False):
        """Creates DogDataset objects, depends on which stage was passed

        Parameters
        ----------
        stage : string (optional)
            Name of the actual stage ('fit' or 'test')
        """

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:

            self.train_split_path, self.val_split_path = train_test_split(
                self.train_path_images, test_size=0.2)

            self.dog_train = DogDataset(self.train_split_path,
                                        self.class_dict,
                                        ImageTransform(self.img_size),
                                        phase='train')

            self.dog_val = DogDataset(self.val_split_path,
                                      self.class_dict,
                                      ImageTransform(self.img_size),
                                      phase='val')

        # Assign enroll dataset for use in dataloader(s)
        if stage == 'test':

            if enroll:
                self.dog_enroll = DogDataset(self.enroll_path_images,
                                             self.enroll_class_dict,
                                             ImageTransform(self.img_size),
                                             phase='test')

            else:
                self.dog_test = DogDataset(self.test_path_images,
                                           self.enroll_class_dict,
                                           ImageTransform(self.img_size),
                                           phase='test')

    def random_plot_enroll_data(self):
        """Randomly pick an image from enroll_path_images to plot in the screen.

        """
        img_path = random.choice(self.enroll_path_images)
        img = Image.open(img_path)
        breed = img_path.split('\\')[0].split('-')[-1]

        print(breed)

        plt.imshow(img)
        plt.axis('off')
        plt.title(breed)
        plt.show()

    def random_plot_train_data(self):
        """Randomly pick an image from train_path_images to plot in the screen.

        """
        img_path = random.choice(self.train_path_images)
        img = Image.open(img_path)
        breed = img_path.split('\\')[0].split('-')[-1]

        print(breed)

        plt.imshow(img)
        plt.axis('off')
        plt.title(breed)
        plt.show()

    def train_dataloader(self):
        return DataLoader(self.dog_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dog_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dog_test, batch_size=self.batch_size, shuffle=False)

    def enroll_dataloader(self):
        return DataLoader(self.dog_enroll, batch_size=self.batch_size, shuffle=False)
