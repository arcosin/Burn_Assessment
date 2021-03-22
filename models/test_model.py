import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import datasets, models, transforms

# Define variables
resnet_input_size = 224
batch_size = 4
num_classes = 2
device = torch.device("cuda:0")
acc = 0.0

# Location of dataset
data_dir = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Thesis_Data\Corrected\Burn_Dataset\HUSD\Test"
weights_path = r"F:\Users\user\Desktop\PURDUE\Research_Thesis\Models\fe_resnet101_std.pth"

## Load data:

# Normalization
preprocess = transforms.Compose([
        transforms.Resize(resnet_input_size),
        transforms.CenterCrop(resnet_input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Create testing datasets
test_burn_dataset = datasets.ImageFolder(data_dir, transform=preprocess)

# Create training and validation dataloaders
test_dataloader = DataLoader(test_burn_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Linear(in_features=num_features, out_features=num_classes)
model.load_state_dict(torch.load(weights_path))
model = model.to(device)
model.eval()

# Testing model

for inputs, labels in test_dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    acc += torch.sum(predictions == labels.data)
print('Testing Accuracy: {:.4f}'.format(acc/len(test_dataloader.dataset)))

