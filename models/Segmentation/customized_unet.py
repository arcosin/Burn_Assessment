import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as f
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from PIL import Image
import random
import json


class BurnDataset(Dataset):

    """
    Class to create our customized dataset
    """

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


# U-net Blocks


class DownConv(nn.Module):

    """
    One Max Pooling
    Two Convolution -> Batch Normalization -> ReLu
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downblock = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.downblock(x)


class UpConv(nn.Module):

    """"
    One up convolution
    Two Convolution -> Batch Normalization -> ReLu
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = self.in_channels // 2
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(self.in_channels, self.in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dif_h = x2.size()[2] - x1.size()[2]
        dif_w = x2.size()[3] - x1.size()[3]
        x1 = f.pad(x1, [dif_w // 2, dif_w - dif_w // 2, dif_h // 2, dif_h - dif_h // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.doubleconv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.doubleconv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Complete model


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.l1 = DoubleConv(self.n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 1024 // factor)
        self.up1 = UpConv(1024, 512 // factor, bilinear)
        self.up2 = UpConv(512, 256 // factor, bilinear)
        self.up3 = UpConv(256, 128 // factor, bilinear)
        self.up4 = UpConv(128, 64, bilinear)
        self.out = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        return logits


def train_model(model, device, epochs, batch_size, lr, n_train, n_val, dataloader):

    start = time.time()

    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    printing_list = []
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-'*30)

        for phase in ['Train', 'Val']:

            running_loss = 0.0
            running_corrects = 0.0

            if phase == 'Train':

                model.train()

                for images, masks in dataloader[phase]:
                    images = images.to(device, dtype=torch.float32)
                    masks = masks.to(device, dtype=torch.long)
                    optimizer.zero_grad()
                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 0.1)
                    optimizer.step()
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(predictions == masks.data) / (256 ** 2)

            else:

                model.eval()

                for images, masks in dataloader[phase]:
                    images = images.to(device, dtype=torch.float32)
                    masks = masks.to(device, dtype=torch.long)
                    optimizer.zero_grad()

                    with torch.no_grad():
                        outputs = model(images)
                        _, predictions = torch.max(outputs, 1)
                        loss = criterion(outputs, masks)
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(predictions == masks.data) / (256 ** 2)

                scheduler.step(running_loss / len(dataloader[phase].dataset))

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = (running_corrects / len(dataloader[phase].dataset)).cpu().numpy()

            print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            printing_list.append('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'Train':
                training_loss.append(epoch_loss)
                training_accuracy.append(epoch_acc)
            else:
                validation_loss.append(epoch_loss)
                validation_accuracy.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())

    train_time = time.time() - start
    print('Training complete in {:.0f}m {:.0f}'.format(train_time // 60, train_time % 60))
    printing_list.append('Training complete in {:.0f}m {:.0f}'.format(train_time // 60, train_time % 60))
    print('Best validation accuracy: {:.4f}'.format(best_acc))
    printing_list.append('Best validation accuracy: {:.4f}'.format(best_acc))

    plt.subplot(1, 2, 1)
    plt.plot(list(range(epochs)), training_loss, color='skyblue', label='Train')
    plt.plot(list(range(epochs)), validation_loss, color='orange', label='Val')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(list(range(epochs)), training_accuracy, color='skyblue', label='Train')
    plt.plot(list(range(epochs)), validation_accuracy, color='orange', label='Val')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    model.load_state_dict(best_model_weights)

    return model, training_loss, training_accuracy, validation_loss, validation_accuracy, printing_list


if __name__ == "__main__":

    # Paths
    data_dir = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Dataset"
    labels_dir = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\RGB\Masks_Greyscale"
    save_dir = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Models\Segmentation\Results_Train_6"

    # Model inputs
    batch_size = 4
    device = torch.device("cuda:0")
    learning_rate = 0.001
    n_epochs = 50
    n_classes = 3
    n_channels = 3

    # Create training and validation datasets
    training_dataset = BurnDataset(os.path.join(data_dir, "Train"), os.path.join(labels_dir, "Train"), train=True)
    val_dataset = BurnDataset(os.path.join(data_dir, "Val"), os.path.join(labels_dir, "Val"), train=False)

    # Create training and validation dataloaders
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    dataloader = {'Train': training_dataloader, 'Val': val_dataloader}

    # Initialize model
    model = UNet(n_channels, n_classes)

    # Training and validation
    segmentation_model, training_loss, training_accuracy, validation_loss, validation_accuracy, printing_list = train_model(model,
                                                device, n_epochs, batch_size, learning_rate, len(training_dataset),
                                                len(val_dataset), dataloader)

    # Save model
    torch.save(segmentation_model.state_dict(), os.path.join(save_dir, "UNet_std.pth"))
    summary_model = {'Training Loss': list(map(str, training_loss)), 'Training Accuracy': list(map(str, training_accuracy)),
                     'Validation Loss': list(map(str, validation_loss)), 'Validation Accuracy': list(map(str, validation_accuracy))}
    json = json.dumps(summary_model)
    file1 = open(os.path.join(save_dir, "summary.txt"), "w")
    file1.write(str(summary_model))
    file1.close()
    file2 = open(os.path.join(save_dir, "summary.json"), "w")
    file2.write(json)
    file2.close()

    file3 = open(os.path.join(save_dir, "print.txt"), "w")
    for lines in printing_list:
        file3.write(lines)
    file3.close()

