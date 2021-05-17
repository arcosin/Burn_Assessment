import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import copy


def initialize_resnet(num_classes, feature_ext):
    'Initialize ResNet18 model'
    model = models.resnet18(pretrained=False)
    if feature_ext:
        for parameter in model.parameters():
            parameter.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(in_features=num_features, out_features=num_classes)
    input_size = 224
    return model, input_size


def create_optimizer(model, feature_ext, lr, momentum):
    'Create optimizer for model'
    params_to_update = model.parameters()
    if feature_ext:
        params_to_update = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad == True:
                params_to_update.append(parameter)
    optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)
    return optimizer


def train_model(model, dataloader, optimizer, num_epochs):
    start = time.time()
    device = torch.device("cuda:0")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Iterate over epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)

        for phase in ['Train', 'Val']:

            if phase == 'Train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, predictions = torch.max(outputs, 1)

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects / len(dataloader[phase].dataset)

            print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'Train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())

        print()

    train_time = time.time() - start
    print('Training complete in {:.0f}m {:.0f}'.format(train_time // 60, train_time % 60))
    print('Best validation accuracy: {:.4f}'.format(best_acc))

    plt.subplot(1, 2, 1)
    plt.plot(list(range(num_epochs)), train_loss, color='skyblue', label='Train')
    plt.plot(list(range(num_epochs)), val_loss, color='orange', label='Val')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(list(range(num_epochs)), train_acc, color='skyblue', label='Train')
    plt.plot(list(range(num_epochs)), val_acc, color='orange', label='Val')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    model.load_state_dict(best_model_weights)

    return model


if __name__ == "__main__":

    # Model inputs
    data_dir = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\Corrected\Burn_Dataset\HUSD"
    save_dir = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Models"
    num_classes = 2
    batch_size = 16
    num_epochs = 30
    feature_extract = False
    learning_rate = 0.001
    momentum = 0.9

    # Initialize model
    pretrained_model, resnet_input_size = initialize_resnet(num_classes, feature_extract)

    # Load data:

    # Data augmentation and normalization
    preprocess = {
        'Train': transforms.Compose([
            transforms.RandomResizedCrop(resnet_input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Val': transforms.Compose([
            transforms.Resize(resnet_input_size),
            transforms.CenterCrop(resnet_input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    train_burn_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Train'), transform=preprocess['Train'])
    val_burn_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Val'), transform=preprocess['Val'])

    # Create training and validation dataloaders
    train_dataloader = DataLoader(train_burn_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_burn_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader = {'Train': train_dataloader, 'Val': val_dataloader}

    # Create optimizer
    optimizer = create_optimizer(pretrained_model, feature_extract, learning_rate, momentum)

    # Training and Validation
    ft_model = train_model(pretrained_model, dataloader, optimizer, num_epochs)

    # Save model
    torch.save(ft_model.state_dict(), os.path.join(save_dir, "fe_resnet101_std.pth"))
    torch.save(ft_model, os.path.join(save_dir, "fe_resnet101.pth"))
