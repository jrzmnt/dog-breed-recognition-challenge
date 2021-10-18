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


# import wandb and login
import wandb
from dataset import DogDataset, ImageTransform, DogDataModule


class DogModel(pl.LightningModule):

    def __init__(self, input_shape, num_classes, learning_rate=2e-4, pre_trained=False):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.pre_trained = pre_trained
        self.params_to_update = []

        print(f'Use Pre-Training? {self.pre_trained}')
        self.feature_extractor = models.resnet50(pretrained=self.pre_trained)

        self.feature_extractor.eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        n_sizes = self._get_conv_output(input_shape)

        self.classifier = nn.Linear(n_sizes, num_classes)

#         # unfreeze the last ResNet layers and freeze all the other
#         if self.pre_trained:
#             update_params_name = ['fc.weight', 'fc.bias']
#             self.params_to_update = []

#             for name, param in self.net.named_parameters():

#                 if name in update_params_name:
#                     print(f'Not going to freeze the layer: {name}')
#                     param.requires_grad = True
#                     self.params_to_update.append(param)

#                 else:
#                     param.requires_grad = False

        # FC Layer should out 100 (num_classes) dog breed classes
        #self.net.fc = nn.Linear(in_features=self.net.fc.in_features, out_features=self.num_classes)

        # set CrossEntropy as the loss
        self.criterion = nn.CrossEntropyLoss()

        # set Adam as the optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)

        # set ExponentialLR as the scheduler
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.99)

    # returns the size of the output tensor going into Linear layer from the conv block.

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x

        # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.classifier(x), dim=1)

        return x

#     # infer x using self.net
#     def forward(self, x):
#         x = self.net(x)
#         return x

    # set the optimizer and the scheduler chosen
    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

        return loss
