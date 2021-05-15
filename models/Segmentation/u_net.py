import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as f
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from PIL import Image
import cv2
import random

#Paths
data_dir = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Dataset"
labels_dir = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Masks_Normalized"

# Customize Dataset Class
class BurnDataset(Dataset):

    def __init__(self, inputs_dir, masks_dir, train=True):
        self.inputs_dir = inputs_dir
        self.masks_dir = masks_dir
        self.data = os.listdir(self.inputs_dir)
        self.train = train

    def __len__(self):
        return len(self.data)

    def preprocess(self, img):
        img_array = np.array(img)
        img_array = img_array.transpose((2, 0, 1))
        if img_array.max() > 1:
            img_array = img_array / 255
        return img_array

    def transform(self, img, mask):
        if self.train:
            if random.random() > 0.5:
                img = f.hflip(img)
                mask = f.hflip(mask)
            if random.random() > 0.5:
                img = f.vflip(img)
                mask = f.vflip(mask)
        return img, mask

    def __getitem__(self, index):
        file_name = self.data[index].split(".")[0]
        input_file = os.path.join(self.inputs_dir, file_name + ".png")
        mask_file = os.path.join(self.masks_dir, file_name + ".png")
        image = Image.open(input_file)
        mask = Image.open(mask_file)
        timage, tmask = self.transform(image, mask)
        image = self.preprocess(timage)
        mask = np.array(tmask) / 255
        im, ground_t = torch.from_numpy(image).type(torch.FloatTensor), torch.from_numpy(mask).type(torch.FloatTensor)
        return im, ground_t

def visualize(img, mask):



if __name__ == "__main__":

    # Model inputs
    batch_size = 4

    # Create training and validation datasets
    training_dataset = BurnDataset(os.path.join(data_dir, "Train"), os.path.join(labels_dir, "Train"))
    val_dataset = BurnDataset(os.path.join(data_dir, "Val"), os.path.join(labels_dir, "Val"))

    # Create training and validation dataloaders
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

